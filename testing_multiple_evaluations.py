import open3d
import trimesh
import skimage.io
import numpy as np
import cadquery as cq
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import glob
import random
from tqdm import tqdm
from multiprocessing import Process

import torch
from torch import nn
from transformers import (
    AutoTokenizer, Qwen2ForCausalLM, Qwen2Model, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

### Define CAD-Recode model

class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.projection = nn.Linear(54, hidden_size)

    def forward(self, points):
        x = points[..., :3]
        x = (x.unsqueeze(-1) * self.frequencies).view(*x.shape[:-1], -1)
        x = torch.cat((points[..., :3], x.sin(), x.cos()), dim=-1)
        x = self.projection(torch.cat((x, points[..., 3:]), dim=-1))
        return x


class CADRecode(Qwen2ForCausalLM):
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        torch.set_default_dtype(torch.float32)
        self.point_encoder = FourierPointEncoder(config.hidden_size)
        torch.set_default_dtype(torch.bfloat16)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                point_cloud=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                cache_position=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # concatenate point and text embeddings
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            assert inputs_embeds is None
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_cloud).bfloat16()
            inputs_embeds[attention_mask == -1] = point_embeds.reshape(-1, point_embeds.shape[2])
            attention_mask[attention_mask == -1] = 1
            input_ids = None
            position_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs['point_cloud'] = kwargs['point_cloud']
        return model_inputs

### Load CAD-Recode checkpoint
def load_cad_recode_checkpoint():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2-1.5B',
        pad_token='<|im_end|>',
        padding_side='left')
    
    model = CADRecode.from_pretrained(
        'filapro/cad-recode',
        torch_dtype='auto',
        attn_implementation=attn_implementation).eval().to(device)

    return tokenizer, model

### Load input point cloud
    
def mesh_to_point_cloud(mesh, n_points=256):
    vertices, faces = trimesh.sample.sample_surface(mesh, n_points)
    point_cloud = np.concatenate((
        np.asarray(vertices),
        mesh.face_normals[faces]
    ), axis=1)
    ids = np.lexsort((point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]))
    point_cloud = point_cloud[ids]
    return point_cloud

def load_input_pc(stl_path, seed=42):
    # gt_mesh = trimesh.load_mesh('./49215_5368e45e_0000.stl')
    gt_mesh = trimesh.load_mesh(stl_path, 'obj')
    gt_mesh.apply_translation(-(gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0)
    gt_mesh.apply_scale(2.0 / max(gt_mesh.extents))
    np.random.seed(seed)
    point_cloud = mesh_to_point_cloud(gt_mesh)

    return point_cloud

### Run CAD-Recode on the input point cloud

def run_cad_recode(tokenizer, model, point_cloud):
    input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [tokenizer('<|im_start|>')['input_ids'][0]]
    attention_mask = [-1] * len(point_cloud) + [1]
    
    with torch.no_grad():
        batch_ids = model.generate(
            input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
            attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
            point_cloud=torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).to(model.device),
            max_new_tokens=768,
            pad_token_id=tokenizer.pad_token_id)
        
    py_string = tokenizer.batch_decode(batch_ids)[0]
    begin = py_string.find('<|im_start|>') + 12
    end = py_string.find('<|endoftext|>')
    py_string = py_string[begin: end]
    
    return py_string

### Execute predicted python code to raise a CAD model

def eval_str(py_string, out_file):
    try:
        exec(py_string, globals())
        compound = globals()['r'].val()
        
        vertices, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        mesh.export(out_file)
    
    except Exception as e:
        print(e)
    
### Compute IoU and Chamfer distance metrics

def compute_chamfer_distance(gt_mesh, pred_mesh):
    pred_mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 2))
    pred_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    gt_mesh.apply_transform(trimesh.transformations.scale_matrix(1 / 2))
    gt_mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))

    n_points = 8192
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))

    intersection_volume = 0
    for gt_mesh_i in gt_mesh.split():
        for pred_mesh_i in pred_mesh.split():
            intersection = gt_mesh_i.intersection(pred_mesh_i)
            volume = intersection.volume if intersection is not None else 0
            intersection_volume += volume

    gt_volume = sum(m.volume for m in gt_mesh.split())
    pred_volume = sum(m.volume for m in pred_mesh.split())
    union_volume = gt_volume + pred_volume - intersection_volume
    iou = intersection_volume / (union_volume + 1e-7)

    print(f'CD: {cd * 1000:.3f}, IoU: {iou:.3f}')
    return cd * 1000, iou

def main():
    tokenizer, model = load_cad_recode_checkpoint()

    root_dir = '/app/test_data'
    num_attempts_per_file = 10
    
    pattern = os.path.join(root_dir, '*')
    file_paths = glob.glob(pattern, recursive=True)
    file_paths = [path for path in file_paths if os.path.isfile(path)]
    
    for file_path in tqdm(file_paths):
        print(f'## Processing file: {file_path} ##')
        base_sample_id = os.path.basename(file_path).split('.')[0]
        results_dir = '/app/test_data_results/' + base_sample_id
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for attempt in tqdm(range(num_attempts_per_file)):
            pc = load_input_pc(stl_path=file_path, seed=attempt*12)

            py_string = run_cad_recode(tokenizer, model, pc)
            out_file = os.path.join(results_dir, f'attempt_{attempt}.stl')
            
            process = Process(target=eval_str, args=(py_string, out_file))
            process.start()
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join()
                
            if os.path.exists(out_file):
                gt_mesh = trimesh.load_mesh(file_path, 'obj')
                gt_mesh.apply_translation(-(gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0)
                gt_mesh.apply_scale(2.0 / max(gt_mesh.extents))
                
                try:
                    pred_mesh = trimesh.load_mesh(out_file)
                    pred_mesh.apply_translation(-(pred_mesh.bounds[0] + pred_mesh.bounds[1]) / 2.0)
                    pred_mesh.apply_scale(2.0 / max(pred_mesh.extents))

                    cd, iou = compute_chamfer_distance(gt_mesh, pred_mesh)

                    with open(out_file.replace('.stl', '.txt'), 'w') as f:
                        f.write(f'CD (x1e3): {cd}\nIoU: {iou}\n\nCode:\n')
                        f.write(py_string)
                        
                except Exception as e:
                    print(e)
                                

if __name__ == '__main__':
    main()