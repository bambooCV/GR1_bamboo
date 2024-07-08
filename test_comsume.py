import torch,clip 
import torch.nn as nn 
import models.vision_transformer as vits
import time
class testmodel(nn.Module):
    def __init__(
        self,
        
    ):
        super().__init__()
        # vision encoders model
        self.model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        # language encoders model
        self.model_clip, _ = clip.load("ViT-B/32",device="cpu") 

            
    def forward(self, rgb_static_norm, language):
        time_start = time.time()
        language_embedding = self.model_clip.encode_text(language).unsqueeze(1)
        obs_embeddings, patch_embeddings = self.model_mae(rgb_static_norm)
        time_end = time.time()
        print(f"Time total cost: {time_end-time_start}")
        
        
        
        # time_start = time.time()
        # language_embedding = self.model_clip.encode_text(language).unsqueeze(1)
        # time_end = time.time()
        # print(f"Time model_clip cost: {time_end-time_start}")

        # time_start = time.time()
        # obs_embeddings, patch_embeddings = self.model_mae(rgb_static_norm)
        # time_end = time.time()
        # print(f"Time model_mae cost: {time_end-time_start}")
        

        return obs_embeddings, patch_embeddings, language_embedding


if __name__ == '__main__':

    model = testmodel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    rgb_static_norm = torch.rand(64,3,224,224).to(device)
    
    language = torch.randint(0,10000,size=(64,77)).to(device)
    
    while True:
        with torch.no_grad():
         
            obs_embeddings, patch_embeddings, language_embedding = model(rgb_static_norm, language)
            time_end = time.time()

    