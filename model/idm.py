import torch
import torch.nn as nn
import transformers
import numpy

'''
IOM must model 
frame at t, frame at t + 1 -> action taken between these frames.
'''

__clip_cache = None
def _get_clip_model():
    global __clip_cache
    if __clip_cache == None:
        #build model
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("Transformers or torch not installed nerd.")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps" #for macbooks.

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # processor has no .to()

        __clip_cache = (device, processor, model)
        print(f"Clip model loaded in cache using device: {device}")

    return __clip_cache  # always return, whether freshly built or already cached


# DEPRECATED: use IDM_FCN instead (window_size=1 is equivalent)
# class IDM_FCN_Single(nn.Module):
#     def __init__(self, clip_cache, embedding_dimension : int , num_action_classes : int):
#         super().__init__()
#         '''
#         You're just storing CLIP as a submodule. PyTorch automatically registers it, meaning its parameters become
#         part of your model's parameter tree. That's why they'd get gradients —
#         and also why freezing them with requires_grad = False works cleanly.
#         '''
#         self.pretrained_clip_model = clip_cache[2]
#         self.pretrained_clip_processor = clip_cache[1] #just preprocessor iage into tensor that s i .
#         self.current_device = clip_cache[0]
#
#         for parameter in self.pretrained_clip_model.parameters():
#             parameter.requires_grad = False #freeze clip weights, so that in training, we do not touch these.
#
#         input_dim = embedding_dimension * 2 #if our window on either side is more than 1 (read 2 frames either
#         #then multiply with window size.)
#
#         self.fcn = nn.Sequential(
#             nn.Linear(in_features=input_dim, out_features=512),
#             nn.ReLU(),
#             nn.Linear(512, num_action_classes)
#         )
#
#         '''
#         oh so in training this would get fucked up since the corss entropy loss module will apply softamz on the logits on its own when trainig? to see if we are close to
#         the anser prob distirbution? so on inference we should osftma that logits outslesv?
#         '''
#
#     def forward(self, frame_t, frame_t_1):
#         #in forward i must call and run clip, with frozen wirghts and then
#         #pass onto the rest of the pipeline.
#         inputs = self.pretrained_clip_processor(images=[frame_t, frame_t_1], return_tensors='pt').to(self.current_device)
#
#         with torch.no_grad():
#             image_embeddings = self.pretrained_clip_model.get_image_features(**inputs)
#
#         #let just concat them for now, perhaps i might want to pool them?
#         #i dont know ? we will see.
#
#         #lets just concat, concat in column dimesnion? not in row since i need (1024,) not (2, 512)
#         fcn_input = image_embeddings.view(1, -1) #esstnially reshape, 1 row and -1 is figure it out yourslef however manh
#         #columns needed.abs
#
#         return self.fcn(fcn_input) #now output would be num actiona classes.



class IDM_FCN(nn.Module):  # renamed from IDM_FCN_Multiple
    def __init__(self, clip_cache, window_size, embedding_dimension : int , num_action_classes : int):
        super().__init__()
        '''
        You're just storing CLIP as a submodule. PyTorch automatically registers it, meaning its parameters become
        part of your model's parameter tree. That's why they'd get gradients —
        and also why freezing them with requires_grad = False works cleanly.
        '''
        self.pretrained_clip_model = clip_cache[2] 
        self.pretrained_clip_processor = clip_cache[1] #just preprocessor iage into tensor that s i .
        self.current_device = clip_cache[0]

        self.window_size = window_size #stores window size of frames on either side.

        for parameter in self.pretrained_clip_model.parameters():
            parameter.requires_grad = False #freeze clip weights, so that in training, we do not touch these.

        input_dim = embedding_dimension * self.window_size * 2  # window frames on each side, all concatenated flat

        self.fcn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(512, num_action_classes)
        )

    def forward(self, frames_t, frames_t_1):
        frames_list = list(frames_t) + list(frames_t_1)  # safe list concat regardless of input type
        inputs = self.pretrained_clip_processor(images=frames_list, return_tensors='pt').to(self.current_device)

        with torch.no_grad():
            image_embeddings = self.pretrained_clip_model.get_image_features(**inputs)

        # flatten all frame embeddings into one vector: (num_frames * embed_dim,)
        fcn_input = image_embeddings.view(1, -1)

        return self.fcn(fcn_input) #now output would be num actiona classes.




