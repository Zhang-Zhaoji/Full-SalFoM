import torch
import os
import torch.nn as nn
from torch.nn.functional import interpolate
from einops import rearrange
import argparse
from einops.layers.torch import Rearrange
from Unmasked_Teacher.single_modality.models.modeling_finetune import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224, vit_large_patch16_384
from Transformer_module import SwinTransformer3D_customized as att_layer


PATH = '----path to ----/l16_ptk710_ftk710_ftk400_f16_res224.pth'




class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners=align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x




class net(nn.Module):
    def __init__(self, pretrained=True):
        super(net, self).__init__()
        
        self.encoder = vit_large_patch16_224(all_frames=16, tubelet_size=1)
        
        #self.encoder.load_state_dict(torch.load(PATH),strict=False)
        


                                ####### 3D conv branch ###########   

        self.conv_1 = nn.Conv3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.conv_2 = nn.Conv3d(512, 256, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_3 = nn.Conv3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.conv_4 = nn.Conv3d(128, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_5 = nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)

        self.conv_6 = nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)

        self.conv_7 = nn.Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_8 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_9 = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)




                                ####### Transformer branch ###########   

        self.transformer_1 = att_layer(patch_size=(2, 1, 1),in_chans=1024,embed_dim=256,depths=[6],num_heads=[8],window_size=(2, 1, 1))
        self.transformer_2 = att_layer(patch_size=(2, 1, 1),in_chans=256,embed_dim=64,depths=[6],num_heads=[8],window_size=(2, 1, 1))
        self.transformer_3 = att_layer(patch_size=(4, 1, 1),in_chans=64,embed_dim=16,depths=[6],num_heads=[8],window_size=(4, 1, 1))

                    

                    
                    
                                ####### 2D conv branch ###########   
        self.conv_t1_1 = nn.Conv3d(1024, 1024, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_t1_2 = nn.Conv3d(1024, 1024, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_t1_3 = nn.Conv3d(1024, 1024, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)
        self.conv_t1_4 = nn.Conv3d(1024, 1024, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False)


        self.conv_t2_1 = nn.Conv3d(256, 256, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_2 = nn.Conv3d(256, 256, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t2_3 = nn.Conv3d(256, 256, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))

        self.conv_t3_1 = nn.Conv3d(64, 64, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_2 = nn.Conv3d(64, 64, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.conv_t3_3 = nn.Conv3d(64, 64, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))




        self.conv_2D_1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_2 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        

        self.conv_2D_4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)



        self.conv_2D_7 = nn.Conv2d(64,32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_8 = nn.Conv2d(32,16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_2D_9 = nn.Conv2d(16,16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)



        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsample_2D = Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU()    
        self.sig = nn.Sigmoid()  
        self.final_map = nn.Conv2d(2, 1, kernel_size=1, stride=1)  

        self.rearrange = Rearrange('b  (t h w) c -> b c t h w', t = 16, h = 14, w=14)

                    
                    
    def forward(self, x):
        

        
        y  = self.encoder(x)
        y = self.rearrange(y)
       
        y_t_1 = self.transformer_1(y)

        y_1 = self.conv_1(y)     
        y_1 = self.relu(y_1)
        
        y_2 = self.conv_2(y_1)     
        y_2 = self.relu(y_2)


        y_t_2 = self.transformer_2(y_t_1[0])
        y_t_2_up = self.upsampling4(y_t_2[0])


        y_2 = y_2 + y_t_1[0]

        y_3 = self.conv_3(y_2)     
        y_3 = self.relu(y_3)
        y_3 = self.upsampling2(y_3)

        
        y_4 = self.conv_4(y_3)     
        y_4 = self.relu(y_4)
        y_4 = self.upsampling2(y_4)


        y_t_3 = self.transformer_3(y_t_2_up)
        y_t_3_up = self.upsampling4(y_t_3[0])

        y_4 = y_4 + y_t_2_up


        y_5 = self.conv_5(y_4)     
        y_5 = self.relu(y_5)
        y_5 = self.upsampling2(y_5)


        y_6 = self.conv_6(y_5)     
        y_6 = self.relu(y_6)
        y_6 = self.upsampling2(y_6)



        y_6 = y_6.view(y_6.size(0), y_6.size(1), y_6.size(3), y_6.size(4))

       
       

        ######## 2D branch ################

        y_2D_t1 = self.conv_t1_1(y)        
        y_2D_t1 = self.relu(y_2D_t1)

        y_2D_t1 = self.conv_t1_2(y_2D_t1)        
        y_2D_t1 = self.relu(y_2D_t1)

        y_2D_t1 = self.conv_t1_3(y_2D_t1)        
        y_2D_t1 = self.relu(y_2D_t1)

        y_2D_t1 = self.conv_t1_4(y_2D_t1)        
        y_2D_t1 = self.relu(y_2D_t1)



        y_2D_1 = self.conv_2D_1(y_2D_t1.squeeze(2))
        y_2D_1 = self.relu(y_2D_1)


        y_2D_2 = self.conv_2D_2(y_2D_1)
        y_2D_2 = self.relu(y_2D_2)        

        y_2D_3 = self.conv_2D_3(y_2D_2)
        y_2D_3 = self.relu(y_2D_3)        



        y_2D_t2 = self.conv_t2_1(y_2)        
        y_2D_t2 = self.relu(y_2D_t2)

        y_2D_t2 = self.conv_t2_2(y_2D_t2)        
        y_2D_t2 = self.relu(y_2D_t2)

        y_2D_t2 = self.conv_t2_3(y_2D_t2)        
        y_2D_t2 = self.relu(y_2D_t2)


        y_2D_3 = y_2D_3 + y_2D_t2.squeeze(2)


        y_2D_4 = self.conv_2D_4(y_2D_3)
        y_2D_4 = self.relu(y_2D_4)        


        y_2D_5 = self.conv_2D_5(y_2D_4)
        y_2D_5 = self.relu(y_2D_5)        
        y_2D_5 = self.upsample_2D(y_2D_5)

        y_2D_6 = self.conv_2D_6(y_2D_5)
        y_2D_6 = self.relu(y_2D_6)        
        y_2D_6 = self.upsample_2D(y_2D_6)


        y_2D_t3 = self.conv_t3_1(y_4)        
        y_2D_t3 = self.relu(y_2D_t3)

        y_2D_t3 = self.conv_t3_2(y_2D_t3)        
        y_2D_t3 = self.relu(y_2D_t3)




        y_2D_6 = y_2D_6 + y_2D_t3.squeeze(2)


        y_2D_7 = self.conv_2D_7(y_2D_6)
        y_2D_7 = self.relu(y_2D_7)        


        y_2D_8 = self.conv_2D_8(y_2D_7)
        y_2D_8 = self.relu(y_2D_8)        
        y_2D_8 = self.upsample_2D(y_2D_8)


        y_2D_9 = self.conv_2D_9(y_2D_8)
        y_2D_9 = self.relu(y_2D_9)        
        y_2D_9 = self.upsample_2D(y_2D_9)



        out = torch.cat((y_2D_9, y_t_3_up.squeeze(2), y_6), 1)


        out = self.conv_7(out)
        out = self.relu(out)

        out = self.conv_8(out)
        out = self.relu(out)

        out = self.conv_9(out)
        out = self.sig(out)


        out = out.view(out.size(0), out.size(2), out.size(3))

        return out
