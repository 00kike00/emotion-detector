import torch
import torch.nn as nn
import torch.nn.functional as F


class ManagerNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        vision_input_dim: int = 7,
        text_input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
    ):
        """
        Late-fusion Manager Network for Multimodal Emotion Recognition.

        Processes a sequence of vision logits via a GRU to build a temporal
        emotional context, then optionally anchors the final decision with
        text logits when available (training and message-send inference).

        Input:
            vision_seq : [B, T, vision_input_dim]  — frame-level logits over time
            text_logits: [B, text_input_dim] | None — utterance-level logits (optional)

        Output:
            logits     : [B, num_classes]
        """
        super(ManagerNet, self).__init__()

        # ── Vision temporal encoder ───────────────────────────────────────────
        # Projects each frame's logits before feeding into GRU
        self.vision_proj = nn.Linear(vision_input_dim, hidden_dim)
        self.vision_bn   = nn.BatchNorm1d(hidden_dim)

        # GRU reads the projected frame sequence and builds a hidden state
        # that accumulates the emotional baseline over time
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # ── Text anchor branch ────────────────────────────────────────────────
        # Projects text logits into the same hidden space as the GRU output
        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.text_bn   = nn.BatchNorm1d(hidden_dim)

        # ── Fusion + classifier ───────────────────────────────────────────────
        # When text IS available  : fuse GRU final state + text projection
        # When text is NOT available: use GRU final state only (zero text)
        # In both cases input to fc1 is hidden_dim * 2 — consistent shape
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn_fc  = nn.BatchNorm1d(hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, num_classes)

    def _encode_vision(self, vision_seq: torch.Tensor) -> torch.Tensor:
        """
        vision_seq: [B, T, vision_input_dim]
        Returns GRU final hidden state: [B, hidden_dim]
        """
        B, T, _ = vision_seq.shape

        # Project each frame: [B*T, vision_input_dim] -> [B*T, hidden_dim]
        proj = self.vision_proj(vision_seq.reshape(B * T, -1))
        proj = F.relu(self.vision_bn(proj))
        proj = proj.reshape(B, T, -1)          # [B, T, hidden_dim]

        _, h_n = self.gru(proj)                # h_n: [num_layers, B, hidden_dim]
        return h_n[-1]                         # top layer final state: [B, hidden_dim]

    def _encode_text(self, text_logits: torch.Tensor) -> torch.Tensor:
        """
        text_logits: [B, text_input_dim]
        Returns projected text features: [B, hidden_dim]
        """
        return F.relu(self.text_bn(self.text_proj(text_logits)))

    def forward(
        self,
        vision_seq: torch.Tensor,
        text_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        vision_seq  : [B, T, vision_input_dim]  — always required
        text_logits : [B, text_input_dim] | None — None during live inference
                      before the user sends a message
        """
        # 1. Encode vision sequence → emotional baseline
        vision_ctx = self._encode_vision(vision_seq)   # [B, hidden_dim]

        # 2. Encode text if available, else zeros (same shape, no gradient)
        if text_logits is not None:
            text_ctx = self._encode_text(text_logits)  # [B, hidden_dim]
        else:
            text_ctx = torch.zeros_like(vision_ctx)    # [B, hidden_dim]

        # 3. Fuse: concatenate vision context + text anchor
        fused = torch.cat([vision_ctx, text_ctx], dim=1)  # [B, hidden_dim * 2]

        # 4. Classify
        features = F.relu(self.bn_fc(self.fc1(self.dropout(fused))))
        logits   = self.fc2(self.dropout(features))

        return logits

    def forward_streaming(
        self,
        frame_logits: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-frame streaming inference for real-time use.
        Call this every frame BEFORE the user sends a message.

        frame_logits : [1, vision_input_dim]   — one frame at a time
        hidden       : [num_layers, 1, hidden_dim] | None — carry between frames

        Returns:
            logits : [1, num_classes]           — current best prediction
            hidden : [num_layers, 1, hidden_dim] — updated hidden state
        """
        # Project single frame
        proj = F.relu(self.vision_bn(self.vision_proj(frame_logits)))  # [1, hidden_dim]
        proj = proj.unsqueeze(1)                                        # [1, 1, hidden_dim]

        # Step GRU by one frame
        _, hidden = self.gru(proj, hidden)     # hidden: [num_layers, 1, hidden_dim]
        vision_ctx = hidden[-1]                # [1, hidden_dim]

        # No text yet — zero anchor
        text_ctx = torch.zeros_like(vision_ctx)
        fused    = torch.cat([vision_ctx, text_ctx], dim=1)

        features = F.relu(self.bn_fc(self.fc1(self.dropout(fused))))
        logits   = self.fc2(self.dropout(features))

        return logits, hidden

    def forward_with_text(
        self,
        frame_logits: torch.Tensor,
        hidden: torch.Tensor,
        text_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Called once when the user sends a message, injecting text logits
        into the already-accumulated hidden state.

        frame_logits : [1, vision_input_dim]        — current frame
        hidden       : [num_layers, 1, hidden_dim]  — state from streaming
        text_logits  : [1, text_input_dim]          — from text expert

        Returns:
            logits : [1, num_classes]               — final anchored prediction
        """
        # One last GRU step with the current frame
        proj = F.relu(self.vision_bn(self.vision_proj(frame_logits)))
        proj = proj.unsqueeze(1)
        _, hidden = self.gru(proj, hidden)
        vision_ctx = hidden[-1]                           # [1, hidden_dim]

        # Inject text anchor
        text_ctx = self._encode_text(text_logits)         # [1, hidden_dim]
        fused    = torch.cat([vision_ctx, text_ctx], dim=1)

        features = F.relu(self.bn_fc(self.fc1(self.dropout(fused))))
        logits   = self.fc2(self.dropout(features))

        return logits