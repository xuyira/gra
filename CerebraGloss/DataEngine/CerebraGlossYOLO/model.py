import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class InceptionBlock1D(nn.Module):
    """
    Channel-independent Inception module, extracts multi-scale features in parallel using convolution kernels of different sizes.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        branch_out_channels = out_channels // 4

        self.branch1 = nn.Conv2d(in_channels, branch_out_channels, kernel_size=(1, 1), stride=(1, stride))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out_channels, kernel_size=(1, 1)), nn.BatchNorm2d(branch_out_channels), nn.ReLU(),
            nn.Conv2d(branch_out_channels, branch_out_channels, kernel_size=(1, 21), stride=(1, stride), padding=(0, 10)) 
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out_channels, kernel_size=(1, 1)), nn.BatchNorm2d(branch_out_channels), nn.ReLU(),
            # Use dilated convolution, better than nn.Conv2d(branch_out_channels, branch_out_channels, kernel_size=(1, 41), stride=(1, stride), padding=(0, 20)) 
            nn.Conv2d(branch_out_channels, branch_out_channels, 
                    kernel_size=(1, 21), 
                    stride=(1, stride), 
                    padding=(0, 20), # padding = (kernel_size - 1) * dilation / 2
                    dilation=(1, 2))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, stride), padding=(0, 2)),
            nn.Conv2d(in_channels, out_channels - 3 * branch_out_channels, kernel_size=(1, 1))
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        '''
        Args:
            x (Tensor[bs,in_channels,C,T]): Input features

        Returns:
            Tensor[bs,out_channels,C,T/stride]: Output features
        '''
        out1 = self.branch1(x) # (bs,branch_out_channels,C,T/stride)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1) # (bs,out_channels,C,T/stride)
        out = self.bn(out)
        return self.relu(out)


class ChannelAttentionInteraction(nn.Module):
    """
    Channel interaction using Transformer Encoder layer.
    Dynamically learns relationships between channels and can handle different numbers of channels via masking.
    """
    def __init__(self, feature_dim, nhead=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor[bs, feature_dim, C, T]): Input features
            pos_embed (Tensor[bs, C, feature_dim]): Channel positional encoding
            pad_mask (Tensor[bs, C]): Attention mask, True means padding and should be ignored

        Returns:
            Tensor[bs, feature_dim, C, T]: Output features
        """
        bs, d_model, num_channels, seq_len = x.shape
        
        # 1. Prepare input
        x_rearranged = x.permute(0, 3, 2, 1).reshape(
            bs * seq_len, num_channels, d_model
        ) # Merge time and batch, use channel as sequence

        # 2. Add positional encoding
        # pos_embed: (bs, c, d) -> (bs, 1, c, d) -> (bs, t, c, d) -> (b*t, c, d)
        pos_embed_expanded = pos_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pos_embed_rearranged = pos_embed_expanded.reshape(
            bs * seq_len, num_channels, d_model
        )
        
        x_with_pos = x_rearranged + pos_embed_rearranged

        # 3. Prepare attention mask
        # pad_mask: (bs, c) -> (bs, 1, c) -> (bs, t, c) -> (b*t, c)
        pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        src_key_padding_mask = pad_mask_expanded.reshape(
            bs * seq_len, num_channels
        )

        # 4. Pass through Transformer Encoder
        output = self.transformer_encoder(x_with_pos, src_key_padding_mask=src_key_padding_mask)

        # 5. Restore original shape
        output_restored = output.reshape(
            bs, seq_len, num_channels, d_model
        ).permute(0, 3, 2, 1)
            
        # 6. Residual connection (x is original input)
        return x + output_restored


class FPN_Backbone(nn.Module):
    """
    Backbone network, now includes channel positional encoding and channel attention module.
    """
    def __init__(self, num_channels=19):
        super().__init__()
        # Stem: (bs, 1, C, 2000) -> (bs, 16, C, 500)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 4), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # [NEW] Channel positional embedding layer, learns a trainable vector for each of 19 standard channels
        self.channel_pos_embedding_64 = nn.Embedding(num_channels, 64)
        self.channel_pos_embedding_128 = nn.Embedding(num_channels, 128)
        self.channel_pos_embedding_256 = nn.Embedding(num_channels, 256)

        # Layer 1 (C3): (bs, 16, C, 500) -> (bs, 64, C, 250)
        self.layer1_inception = InceptionBlock1D(16, 64, stride=2)
        self.layer1_attn = ChannelAttentionInteraction(64)
        
        # Layer 2 (C4): (bs, 64, C, 250) -> (bs, 128, C, 125)
        self.layer2_inception = InceptionBlock1D(64, 128, stride=2)
        self.layer2_attn = ChannelAttentionInteraction(128)

        # Layer 3 (C5): (bs, 128, C, 125) -> (bs, 256, C, 63)
        self.layer3_inception = InceptionBlock1D(128, 256, stride=2)
        self.layer3_attn = ChannelAttentionInteraction(256)

        # Layer 4 (C6): (bs, 256, C, 63) -> (bs, 512, C, 32)
        self.layer4 = nn.Sequential(InceptionBlock1D(256, 512, stride=2))
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, pos_indices: torch.Tensor):
        """
        Receives extra mask and position indices.
        Args:
            x (Tensor[bs, 1, C, T]): Input features
            attention_mask (Tensor[bs, C]): Attention mask
            pos_indices (Tensor[bs, C]): Channel position indices
        Returns:
            List[Tensor]: List of feature maps from different levels
        """
        # --- Get positional encoding ---
        pos_embed_64 = self.channel_pos_embedding_64(pos_indices)
        pos_embed_128 = self.channel_pos_embedding_128(pos_indices)
        pos_embed_256 = self.channel_pos_embedding_256(pos_indices)

        # --- Forward propagation ---
        c1 = self.stem(x)
        
        c3_temp = self.layer1_inception(c1)
        c3 = self.layer1_attn(c3_temp, pos_embed_64, attention_mask)
        
        c4_temp = self.layer2_inception(c3)
        c4 = self.layer2_attn(c4_temp, pos_embed_128, attention_mask)
        
        c5_temp = self.layer3_inception(c4)
        c5 = self.layer3_attn(c5_temp, pos_embed_256, attention_mask)
        
        c6 = self.layer4(c5)
        
        return [c3, c4, c5, c6]


class FPN_Neck(nn.Module):
    """
    FPN neck, fuses features from Backbone and generates P3, P4, P5, P6
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        '''
        Args:
            in_channels_list (List[int]): List of feature dimensions output by Backbone
            out_channels (int): Each feature dimension is scaled by 1x1 convolution
        '''
        super().__init__()
        # Dynamically create lateral connection layers based on input channel list
        self.lat_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        # Dynamically create subsequent convolution layers
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1)) for _ in in_channels_list])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Receive C3, C4, C5, C6 from Backbone
        c3, c4, c5, c6 = inputs
        
        # Lateral Connections
        p6_lat = self.lat_convs[3](c6)
        p5_lat = self.lat_convs[2](c5)
        p4_lat = self.lat_convs[1](c4)
        p3_lat = self.lat_convs[0](c3)

        # Top-down Path
        p5 = p5_lat + F.interpolate(p6_lat, size=c5.shape[2:], mode='nearest')
        p4 = p4_lat + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = p3_lat + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        
        # Final output layers, processed by convolution to eliminate aliasing effects from upsampling
        p6 = self.fpn_convs[3](p6_lat)
        p5 = self.fpn_convs[2](p5)
        p4 = self.fpn_convs[1](p4)
        p3 = self.fpn_convs[0](p3)
        
        return [p3, p4, p5, p6]
    

class PredictionHead(nn.Module):
    """
    A simple prediction head, converts FPN feature maps to YOLO prediction format
    """
    def __init__(self, in_channels: int, num_classes: int, num_anchors_per_level: int):
        '''
        Args:
            in_channels (int): Input dimension
            num_classes (int): Number of classes
            num_anchors_per_level (int): Number of anchors per grid
        '''
        super().__init__()
        output_size = num_anchors_per_level * (3 + num_classes) # 3 = tx,tw,conf
        self.conv = nn.Conv2d(in_channels, output_size, kernel_size=1)
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_level

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x (Tensor[bs,in_channels,C,S_level])

        Returns:
            x (Tensor[bs,C,S_level,B_level,3+num_class])
        '''
        x = self.conv(x) # (bs, B*(3+num_cls), C, S_level)
        batch, _, channels, steps = x.shape
        x = x.reshape(
            batch,
            self.num_anchors,
            3 + self.num_classes,
            channels,
            steps,
        ).permute(0, 3, 4, 1, 2) # (bs, C, S_level, B_level, 3+num_classes)
        t_x = torch.sigmoid(x[..., 0:1])
        t_w = x[..., 1:2]
        confidence = x[..., 2:3]
        class_logits = x[..., 3:]
        
        return torch.cat([t_x, t_w, confidence, class_logits], dim=-1)
    
    
class CerebraGlossYOLO(nn.Module):
    def __init__(self, num_classes: int = 11, 
                 num_anchors_per_level: List[Optional[int]] = [2, None, None, 1], 
                 num_std_channels: int = 19):
        """
        CerebraGlossYOLO, A channel-wise detector for raw EEG waveform detection

        Args:
            num_classes (int): Number of target classes.
            num_anchors_per_level (List[Optional[int]]): 
                A list of 4 elements, corresponding to FPN's P3, P4, P5, P6 layers.
                Each element can be an integer (number of anchors for that layer) or None (disable prediction for that layer).
                Example: [None, 3, 3, None] means only predict on P4 and P5, each with 3 anchors.
            num_std_channels (int): Number of standard EEG channels.
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Record which levels are active
        self.active_levels = [i for i, num in enumerate(num_anchors_per_level) if num is not None]
        print(f"CerebraGlossYOLO initialized. Active prediction levels: {[f'P{i+3}' for i in self.active_levels]}")

        # 1. Backbone network
        self.backbone = FPN_Backbone(num_channels=num_std_channels)
        
        # 2. Neck network
        fpn_in_channels = [64, 128, 256, 512]
        fpn_out_channels = 128
        self.neck = FPN_Neck(fpn_in_channels, fpn_out_channels)
        
        # 3. Prediction heads (key modification)
        # Only create prediction heads for non-None levels
        self.prediction_heads = nn.ModuleList()
        for num_anchors in num_anchors_per_level:
            if num_anchors is not None:
                head = PredictionHead(fpn_out_channels, num_classes, num_anchors)
                self.prediction_heads.append(head)
            else:
                # To keep indices aligned, you could add a None, but a cleaner way is dynamic mapping
                # Here we choose to map dynamically in forward, ModuleList only stores actual modules
                pass
        
        # Create mapping from active level index to prediction head index
        # For example, if active_levels = [1, 2], then P4 (index 1) maps to head 0, P5 (index 2) maps to head 1
        self.level_to_head_map = {level_idx: head_idx for head_idx, level_idx in enumerate(self.active_levels)}


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, pos_indices: torch.Tensor) -> List[torch.Tensor]:
        """
        Receives and passes mask and position indices.

        Args:
            x (Tensor[bs,C,T]): Input signal
            attention_mask (Tensor[bs,C]): Attention mask, True means the channel is zero-padded
            pos_indices (Tensor[bs,C]): Channel position indices, channel order corresponds to standard 10-20 system

        Returns:
            List[Tensor[bs,C,S_level,B_level,3+num_class]]: List containing outputs only from activated FPN levels
        """
        x = x.unsqueeze(1) # (bs, 1, C, T)
        
        # Backbone and Neck computation remains unchanged, always generates P3, P4, P5, P6
        c3, c4, c5, c6 = self.backbone(x, attention_mask, pos_indices)
        p_outputs = self.neck([c3, c4, c5, c6]) # p_outputs is [p3, p4, p5, p6]

        # (Key modification) Only predict on activated levels
        predictions = []
        for level_idx in self.active_levels:
            # Get corresponding feature map
            feature_map = p_outputs[level_idx]
            # Get corresponding prediction head
            head_idx = self.level_to_head_map[level_idx]
            head = self.prediction_heads[head_idx]
            # Make prediction and add to result list
            predictions.append(head(feature_map))
        return predictions
