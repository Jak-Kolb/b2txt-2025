import torch 
from torch import nn
from torch.utils.checkpoint import checkpoint

class GRUDecoder(nn.Module):
    '''
    Defines the GRU decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
    '''
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0,
                 bidirectional = False,
                 lookahead_frames = 0,
                 lookahead_units = 0,
                 window_past_frames = None,
                 window_future_frames = None,
                 ):
        '''
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes 
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers 
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step 
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input 
        bidirectional (bool)   - run the GRU in both directions over the whole trial (offline only; cannot stream)
        lookahead_frames (int) - right context R: the output for frame t also sees the causal GRU outputs for frames t+1..t+R
        lookahead_units (int)  - hidden size of the look-ahead GRU; 0 disables the module (R = 0 with units > 0 is its
                                 capacity control: same parameters, no future frames)
        window_past_frames / window_future_frames (int) - sliding-window bidirectional GRU (needs bidirectional):
                                 the output for frame t comes from running the bidirectional GRU over frames
                                 t-P..t+R only, restarted at every frame; nothing outside the window is remembered
        '''
        super(GRUDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lookahead_frames = lookahead_frames
        self.lookahead_units = lookahead_units
        if lookahead_units > 0 and bidirectional:
            raise ValueError("The look-ahead module sits on a causal GRU; it cannot be combined with bidirectional")
        if lookahead_frames < 0 or (lookahead_frames > 0 and lookahead_units == 0):
            raise ValueError("lookahead_frames needs lookahead_units > 0 (and must be >= 0)")

        self.window_past_frames = window_past_frames
        self.window_future_frames = window_future_frames
        self.windowed = window_past_frames is not None or window_future_frames is not None
        if self.windowed and (not bidirectional or window_past_frames is None or window_future_frames is None
                              or window_past_frames < 0 or window_future_frames < 0):
            raise ValueError("A sliding window needs bidirectional = True and both window sizes >= 0")
        # Models whose outputs near a trial's end depend on later frames need each trial's length in padded batches
        self.uses_frame_lens = lookahead_units > 0 or self.windowed

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size = self.input_size,
            hidden_size = self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = self.bidirectional,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Optional bounded look-ahead: a backward GRU over the causal outputs of frames t..t+R, run from t+R down to t
        if self.lookahead_units > 0:
            self.lookahead_gru = nn.GRU(input_size = self.n_units, hidden_size = self.lookahead_units, batch_first = True)
            for name, param in self.lookahead_gru.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units * self.n_directions + self.lookahead_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states = None, return_state = False, frame_lens = None):
        '''
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x. 
        frame_lens (tensor) - (look-ahead models, padded batches) number of valid output frames per trial; frames past a
                              trial's end are zeroed before the look-ahead window, as when decoding that trial alone
        '''

        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0: 
  
            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]
            
            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]
            
            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        
        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers * self.n_directions, x.shape[0], self.n_units).contiguous()

        if self.windowed:
            logits = self.out(self._window_gru(x, frame_lens))
            return (logits, None) if return_state else logits

        # Pass input through RNN 
        output, hidden_states = self.gru(x, states)

        if self.lookahead_units > 0:
            output = torch.cat([output, self._lookahead(output, frame_lens)], dim = -1)

        # Compute logits
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits

    def _lookahead(self, h, frame_lens = None):
        '''Backward GRU over h[t..t+R] for every frame t, frames past the end (or past frame_lens) counting as zeros.

        Equivalent to running self.lookahead_gru on each window h[t+R], ..., h[t] and keeping its final state, but
        steps all frames in parallel (one recurrence step per offset) instead of materializing R+1 copies per frame.
        '''
        n_frames = h.shape[1]
        r = self.lookahead_frames
        if frame_lens is not None:
            valid = torch.arange(n_frames, device = h.device).unsqueeze(0) < frame_lens.to(h.device).unsqueeze(1)
            h = h * valid.unsqueeze(-1).to(h.dtype)
        h = nn.functional.pad(h, (0, 0, 0, r))                    # [batch, frames + R, units]
        gru = self.lookahead_gru                                  # gate order r, z, n as in nn.GRU
        x_gates = nn.functional.linear(h, gru.weight_ih_l0, gru.bias_ih_l0)
        state = None
        for offset in range(r, -1, -1):                           # frame t + offset, from t+R down to t
            xr, xz, xn = x_gates[:, offset:offset + n_frames].chunk(3, dim = -1)
            if state is None:
                state = torch.zeros_like(xn)
            hr, hz, hn = nn.functional.linear(state, gru.weight_hh_l0, gru.bias_hh_l0).chunk(3, dim = -1)
            reset = torch.sigmoid(xr + hr)
            update = torch.sigmoid(xz + hz)
            state = (1 - update) * torch.tanh(xn + reset * hn) + update * state
        return state

    def _window_gru(self, x, frame_lens = None):
        '''Bidirectional self.gru run separately over frames t-P..t+R for every output frame t; returns [batch, frames, 2H].

        Frames before the trial start, past its end or past frame_lens are zero inputs. All windows are stepped in
        parallel (one recurrence step per window position); the first layer's input projection is computed once per
        frame and shared by the windows that contain it. Each layer is recomputed in the backward pass to bound memory.
        '''
        n_frames = x.shape[1]
        past, future = self.window_past_frames, self.window_future_frames
        if frame_lens is not None:
            valid = torch.arange(n_frames, device = x.device).unsqueeze(0) < frame_lens.to(x.device).unsqueeze(1)
            x = x * valid.unsqueeze(-1).to(x.dtype)
        y = nn.functional.pad(x, (0, 0, past, future))           # window position k of frame t is padded frame t+k
        for layer in range(self.n_layers):
            if self.training and torch.is_grad_enabled():
                y = checkpoint(self._window_layer, y, layer, n_frames, use_reentrant = False)
            else:
                y = self._window_layer(y, layer, n_frames)
        return y[:, :, past]                                     # each window's output at its own frame t

    def _window_layer(self, inputs, layer, n_frames):
        '''One bidirectional layer over all windows: inputs are padded frames (layer 0) or [batch, frames, W, 2H].'''
        width = self.window_past_frames + self.window_future_frames + 1
        outputs = []
        for suffix, positions in (("", range(width)), ("_reverse", range(width - 1, -1, -1))):
            w_ih, b_ih = getattr(self.gru, f"weight_ih_l{layer}{suffix}"), getattr(self.gru, f"bias_ih_l{layer}{suffix}")
            w_hh, b_hh = getattr(self.gru, f"weight_hh_l{layer}{suffix}"), getattr(self.gru, f"bias_hh_l{layer}{suffix}")
            x_gates = nn.functional.linear(inputs, w_ih, b_ih)
            state = self.h0.to(x_gates.dtype).expand(inputs.shape[0], n_frames, self.n_units)
            states = [None] * width
            for k in positions:
                gates = x_gates[:, k:k + n_frames] if layer == 0 else x_gates[:, :, k]
                xr, xz, xn = gates.chunk(3, dim = -1)
                hr, hz, hn = nn.functional.linear(state, w_hh, b_hh).chunk(3, dim = -1)
                reset = torch.sigmoid(xr + hr)
                update = torch.sigmoid(xz + hz)
                state = (1 - update) * torch.tanh(xn + reset * hn) + update * state
                states[k] = state
            outputs.append(torch.stack(states, dim = 2))           # [batch, frames, W, H]
        y = torch.cat(outputs, dim = -1)                          # [batch, frames, W, 2H]
        if layer < self.n_layers - 1 and self.rnn_dropout > 0:
            y = nn.functional.dropout(y, p = self.rnn_dropout, training = self.training)
        return y
