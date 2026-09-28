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
                          n_layers=model_args['n_layers'], **common)
    if model_type == 'transformer':
        from transformer_model import TransformerDecoder
        t = model_args['transformer']
        return TransformerDecoder(d_model=t['d_model'], n_layers=t['n_layers'], n_heads=t['n_heads'],
                                  ffn_mult=t['ffn_mult'], dropout=t['dropout'], window=t['window'], **common)
    raise ValueError(f"Unknown model.type: {model_type}")
