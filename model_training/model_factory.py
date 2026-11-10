from rnn_model import GRUDecoder


def build_model(args):
    '''Build the acoustic model named by args.model.type (default: the GRU).'''
    model_args = args['model']
    common = dict(
        neural_dim=model_args['n_input_features'],
        n_days=len(args['dataset']['sessions']),
        n_classes=args['dataset']['n_classes'],
        input_dropout=model_args['input_network']['input_layer_dropout'],
        patch_size=model_args['patch_size'],
        patch_stride=model_args['patch_stride'],
    )
    model_type = model_args.get('type', 'gru')
    if model_type == 'gru':
        return GRUDecoder(n_units=model_args['n_units'], rnn_dropout=model_args['rnn_dropout'],
                          n_layers=model_args['n_layers'], bidirectional=model_args.get('bidirectional', False),
                          lookahead_frames=model_args.get('lookahead_frames', 0),
                          lookahead_units=model_args.get('lookahead_units', 0),
                          window_past_frames=model_args.get('window_past_frames'),
                          window_future_frames=model_args.get('window_future_frames'),
                          input_bottleneck=model_args.get('input_bottleneck', 0),
                          lookahead_input=model_args.get('lookahead_input', 'hidden'),
                          lookahead_raw_dim=model_args.get('lookahead_raw_dim', 256),
                          lookahead_mode=model_args.get('lookahead_mode', 'backward'), **common)
    if model_type == 'transformer':
        from transformer_model import TransformerDecoder
        t = model_args['transformer']
        return TransformerDecoder(d_model=t['d_model'], n_layers=t['n_layers'], n_heads=t['n_heads'],
                                  ffn_mult=t['ffn_mult'], dropout=t['dropout'], window=t['window'],
                                  mask_token=t.get('mask_token', False),
                                  lookahead_frames=t.get('lookahead_frames', 0),
                                  block_type=t.get('block_type', 'standard'), conv_kernel=t.get('conv_kernel', 9),
                                  drop_path=t.get('drop_path', 0.0), shared_input=t.get('shared_input', False),
                                  inter_ctc_layer=t.get('inter_ctc_layer', 0),
                                  no_day_input=t.get('no_day_input', False),
                                  pos_bias=t.get('pos_bias', 'alibi'), window_min=t.get('window_min', 0),
                                  embed_dropout=t.get('embed_dropout'), ffn_dropout=t.get('ffn_dropout', False),
                                  init=t.get('init', 'xavier'), **common)
    raise ValueError(f"Unknown model.type: {model_type}")
