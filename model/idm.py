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
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_action_classes)
        ).to(self.current_device)  # move fcn to same device as clip, otherwise weights will be on cpu and inputs on mps/cuda.

    def forward(self, frames_t, frames_t_1):
        frames_list = list(frames_t) + list(frames_t_1)  # safe list concat regardless of input type
        inputs = self.pretrained_clip_processor(images=frames_list, return_tensors='pt').to(self.current_device)

        with torch.no_grad():
            # call vision model directly and grab pooler_output — more explicit than get_image_features
            # which returns different types depending on transformers version. then project into
            # clip embedding space the same way get_image_features does internally.
            vision_outputs  = self.pretrained_clip_model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
            image_embeddings = self.pretrained_clip_model.visual_projection(vision_outputs.pooler_output)

        # flatten all frame embeddings into one vector: (num_frames * embed_dim,)
        fcn_input = image_embeddings.view(1, -1)

        return self.fcn(fcn_input) #now output would be num actiona classes.


class IDM_Transformer(nn.Module):
    def __init__(self, clip_cache, window_size_t, embedding_dimension : int , num_action_classes : int):
        super().__init__()

        self.pretrained_clip_model = clip_cache[2] 
        self.pretrained_clip_processor = clip_cache[1] #just preprocessor iage into tensor that s i .
        self.current_device = clip_cache[0]

        self.window_size_t = window_size_t
        self.hidden_dim = embedding_dimension
        self.num_action_classes = num_action_classes

        for parameter in self.pretrained_clip_model.parameters():
            parameter.requires_grad = False

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.window_size_t, self.hidden_dim)
            ).to(self.current_device)
        
        #we dont need any input dimensions, since a transfomer expetcs a matrix X of (BATCHSIZE, WINDOW SIZE, EMBEDDING SIZE)
        #so all i need to do in forward is go ove rframe batch take each frame out encode it using clip, and assemble the entire 
        #matrix and send it direct to the stupid trasnfomer encoder. (no decoder since no need for masking, we need to do unmaked traiing seq2seq operation
        #not an autoregresive opration like lanaguge egenration)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead= 8, #8 attention heads.
            dim_feedforward= 4 * self.hidden_dim,
            batch_first=True
        ).to(self.current_device)

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers = 2
        ).to(self.current_device)

        #this is standard mlp, need to apply mlp for each window. pytorch will do it automatically?
        #yes pytroch needs to run on each h latent embeddings (each trasnfomed frame embedding) will need to be 
        #used to cacluate action (caslificaiton task) instead of foing a concat and then push, we are 
        #trasnfome and then classify each frame.

        #its actuall better to use a pairwie head.. (ht, ht+1 and then diff bw ht+1 and ht) the diff is our action signal
        #what changed between these two frames.
        #so then input dims are 512 + 512 + 512 (three embeddings)
        #so final shape of matrix that will go into head is b, t-1, 1536
        self.action_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_action_classes),
        ).to(self.current_device)  # must be on same device as transformer output.

        self.action_head_no_pair = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_action_classes),
        ).to(self.current_device)  # must be on same device as transformer output.

    @property
    def device(self):
        # alias so train_transformer.py can use model.device consistently.
        return self.current_device

    def forward(self, frame_batch):
        '''
        frames_batch:  list of B windows, each window is a list of T frames
        frames batch is just a python list of b windows (batch size, 32) and t frames in each window.
        '''

        frames_batch_embedded = []
        for window_frames in frame_batch:
            # window_frames is a list of T raw numpy (H,W,C) arrays — need to run through
            # clip processor first to get pixel_values tensor before vision model can use them.
            inputs = self.pretrained_clip_processor(images=window_frames, return_tensors='pt').to(self.current_device)
            with torch.no_grad():
                vision_outputs   = self.pretrained_clip_model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
                image_embeddings = self.pretrained_clip_model.visual_projection(vision_outputs.pooler_output)
                # output is (T, 512) — one embedding per frame in this window.
            frames_batch_embedded.append(image_embeddings)

        x = torch.stack(frames_batch_embedded, dim=0).to(self.current_device)
        x = x + self.pos_embedding[:, :x.size(1), :] #append positonal embedding
        #final shape should be batch, window, embedding space.

        #now direct push this into trasnfomer,.
        latent_space_output = self.transformer(x)

        #generate pair wise embeddings
        pair_repr = torch.cat([ #and concat everything exapnd basically.
            latent_space_output[:, :-1, :],              # all tensors expect last one 
            latent_space_output[:, 1:, :],               # all tensors after first shifted
            latent_space_output[:, 1:, :] - latent_space_output[:, :-1, :] # difference
        ], dim=-1)

        #logits = self.action_head(pair_repr)

        logits = self.action_head_no_pair(latent_space_output)
        return logits[:, :-1]

        #return logits




