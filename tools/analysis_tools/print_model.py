# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from collections import OrderedDict

import torch
from mmengine.config import Config, DictAction
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.utils import digit_version
from torchviz import make_dot

from mmdet.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Print model structure and parameters')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device used for calculating')
    parser.add_argument(
        '--show-params',
        action='store_true',
        help='show parameters of each layer')
    parser.add_argument(
        '--show-computation-graph',
        action='store_true',
        help='show computation graph')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='directory to save visualization results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def get_model_complexity_info(model, input_shape, show_params=False):
    """Get model complexity information.

    Args:
        model (nn.Module): The model to analyze.
        input_shape (tuple): Input shape of the model.
        show_params (bool): Whether to show parameters of each layer.

    Returns:
        dict: Model complexity information.
    """
    if torch.cuda.is_available():
        model = model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()

    # Prepare input tensor
    if len(input_shape) == 1:
        input_shape = (1, 3, input_shape[0], input_shape[0])
    elif len(input_shape) == 2:
        input_shape = (1, 3, input_shape[0], input_shape[1])
    else:
        raise ValueError('input shape should be either (H,) or (H, W)')

    input_tensor = torch.randn(input_shape).cuda()

    # Get model structure
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    def get_layer_info(module, prefix=''):
        layer_info = OrderedDict()
        for name, child in module.named_children():
            layer_name = f'{prefix}.{name}' if prefix else name
            layer_info[layer_name] = {
                'type': child.__class__.__name__,
                'params': count_parameters(child),
                'children': get_layer_info(child, layer_name) if len(list(child.children())) > 0 else None
            }
        return layer_info

    model_info = {
        'name': model.__class__.__name__,
        'input_shape': input_shape,
        'total_params': count_parameters(model),
        'layers': get_layer_info(model)
    }

    # Print model structure
    print('\nModel Structure:')
    print('=' * 50)
    print(f'Model: {model_info["name"]}')
    print(f'Input shape: {input_shape}')
    print(f'Total parameters: {model_info["total_params"]:,}')
    print('=' * 50)

    def print_layer_info(layer_info, indent=0):
        for name, info in layer_info.items():
            print(' ' * indent + f'{name}:')
            print(' ' * (indent + 2) + f'Type: {info["type"]}')
            if show_params:
                print(' ' * (indent + 2) + f'Parameters: {info["params"]:,}')
            if info['children']:
                print_layer_info(info['children'], indent + 2)

    print_layer_info(model_info['layers'])
    print('=' * 50)

    return model_info


def visualize_computation_graph(model, input_tensor, output_dir):
    """Visualize model computation graph.

    Args:
        model (nn.Module): The model to visualize.
        input_tensor (torch.Tensor): Input tensor for the model.
        output_dir (str): Directory to save the visualization.
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Get model output and intermediate features
    with torch.no_grad():
        # Register forward hooks to get intermediate features
        features = []
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                features.append((module.__class__.__name__, output))
            elif isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        features.append((f"{module.__class__.__name__}.output_{i}", out))
            elif isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        features.append((f"{module.__class__.__name__}.{k}", v))
        
        # Register hooks on all major components
        hooks = []
        for name, module in model.named_modules():
            # Skip very small modules to avoid too many hooks
            if sum(p.numel() for p in module.parameters()) > 1000:
                hooks.append(module.register_forward_hook(hook_fn))
                print(f"Registered hook for {name}")
        
        # Forward pass
        print("\nRunning forward pass to collect features...")
        _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"\nFound {len(features)} potential tensors for visualization:")
        for i, (name, feat) in enumerate(features):
            print(f"{i}: {name} - shape: {feat.shape}")
        
        # Use the last intermediate feature for visualization
        if features:
            # Try to find a feature with reasonable size (not too small, not too large)
            suitable_features = []
            for name, feat in features:
                if len(feat.shape) >= 2 and feat.numel() < 1e7:  # Avoid extremely large tensors
                    suitable_features.append((name, feat))
            
            if suitable_features:
                # Use the feature with the most channels (typically more informative)
                output_name, output = max(suitable_features, key=lambda x: x[1].shape[1])
                print(f"\nUsing tensor from {output_name} for visualization")
            else:
                # Fallback to the last feature
                output_name, output = features[-1]
                print(f"\nNo suitable feature found, using last tensor from {output_name}")
        else:
            # If no intermediate features found, try to use the output
            print("\nNo intermediate features found, trying model output...")
            output = model(input_tensor)
            
            def find_tensor(obj, path=""):
                if isinstance(obj, torch.Tensor):
                    return [(path, obj)]
                elif isinstance(obj, (list, tuple)):
                    tensors = []
                    for i, item in enumerate(obj):
                        tensors.extend(find_tensor(item, f"{path}[{i}]"))
                    return tensors
                elif isinstance(obj, dict):
                    tensors = []
                    for k, v in obj.items():
                        tensors.extend(find_tensor(v, f"{path}.{k}" if path else k))
                    return tensors
                return []
            
            tensors = find_tensor(output)
            if tensors:
                print("\nFound tensors in model output:")
                for i, (name, tensor) in enumerate(tensors):
                    print(f"{i}: {name} - shape: {tensor.shape}")
                
                # Use the tensor with the most channels
                output_name, output = max(tensors, key=lambda x: x[1].shape[1] if len(x[1].shape) > 1 else 0)
                print(f"\nUsing tensor from {output_name} for visualization")
            else:
                raise ValueError('Could not find suitable tensor for visualization. '
                               'The model output and intermediate features are not tensors.')

    # Create computation graph
    print("\nGenerating computation graph...")
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # Save graph
    graph_path = osp.join(output_dir, 'computation_graph')
    dot.render(graph_path, format='png', cleanup=True)
    print(f'\nComputation graph has been saved to {graph_path}.png')


def main():
    args = parse_args()

    if digit_version(torch.__version__) < digit_version('1.12'):
        print('Warning: Some config files may have compatibility issues with '
              'torch.jit when torch<1.12. Please make sure your pytorch '
              'version is >=1.12.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Build model
    model = MODELS.build(cfg.model)
    
    # Get model complexity info
    model_info = get_model_complexity_info(
        model, args.shape, show_params=args.show_params)

    # Visualize computation graph if needed
    if args.show_computation_graph:
        if torch.cuda.is_available():
            model = model.cuda()
        input_tensor = torch.randn(
            (1, 3, args.shape[0], args.shape[1])).cuda()
        visualize_computation_graph(
            model, input_tensor, args.output_dir or cfg.work_dir)


if __name__ == '__main__':
    main() 