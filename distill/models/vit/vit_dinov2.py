import os 
import torch 

dinov2_path = os.path.join(__file__, "../../../ext/dinov2")
dinov2_vitg14_reg = torch.hub.load(os.path.abspath(dinov2_path), 'dinov2_vitg14_reg', source="local")
dinov2_vitg14_reg = dinov2_vitg14_reg

if __name__ == "__main__":
    with torch.no_grad():
        x = torch.rand((1, 3, 588, 966)).to('cuda')
        model = dinov2_vitg14_reg.to('cuda')
        result = model.get_intermediate_layers(x, [9, 19, 29, 39]) # get intermedia
        import pdb; pdb.set_trace()