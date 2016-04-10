"""
build_model.py

Build entire memory network from individual layers. Takes a bAbI Configuration as input.
"""
import numpy as np


def build_model(general_config):
    """
    Build model

    NOTE: (for default config)
    1) Model's architecture (embedding B)
        LookupTable -> ElemMult -> Sum ->
                                  [ Duplicate -> { Parallel -> Memory -> Identity } -> AddTable ]
                                  -> LinearNB -> Softmax
    2) Memory's architecture
        a) Query module (embedding A)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd -> Softmax
        b) Output module (embedding C)
            Parallel -> { LookupTable + ElemMult + Sum } -> Identity -> MatVecProd
    """
    train_config = general_config.train_config
    dictionary = general_config.dictionary
    use_bow = general_config.use_bow
    nhops = general_config.nhops
    add_proj = general_config.add_proj
    share_type = general_config.share_type
    enable_time = general_config.enable_time
    add_nonlin = general_config.add_nonlin

    in_dim = train_config["in_dim"]
    out_dim = train_config["out_dim"]
    max_words = train_config["max_words"]
    voc_sz = train_config["voc_sz"]

    if not use_bow:
        train_config["weight"] = np.ones((in_dim, max_words), np.float32)
        for i in range(in_dim):
            for j in range(max_words):
                train_config["weight"][i][j] = (i + 1 - (in_dim + 1) / 2) * \
                                               (j + 1 - (max_words + 1) / 2)
        train_config["weight"] = \
            1 + 4 * train_config["weight"] / (in_dim * max_words)

    memory = {}
    model = Sequential()
    model.add(LookupTable(voc_sz, in_dim))
    if not use_bow:
        if enable_time:
            model.add(ElemMult(train_config["weight"][:, :-1]))
        else:
            model.add(ElemMult(train_config["weight"]))

    model.add(Sum(dim=1))

    proj = {}
    for i in range(nhops):
        if use_bow:
            memory[i] = MemoryBoW(train_config)
        else:
            memory[i] = MemoryL(train_config)

        # Override nil_word which is initialized in "self.nil_word = train_config["voc_sz"]"
        memory[i].nil_word = dictionary['nil']
        model.add(Duplicate())
        p = Parallel()
        p.add(memory[i])

        if add_proj:
            proj[i] = LinearNB(in_dim, in_dim)
            p.add(proj[i])
        else:
            p.add(Identity())

        model.add(p)
        model.add(AddTable())
        if add_nonlin:
            model.add(ReLU())

    model.add(LinearNB(out_dim, voc_sz, True))
    model.add(Softmax())

    # Share weights
    if share_type == 1:
        # Type 1: adjacent weight tying
        memory[0].emb_query.share(model.modules[0])
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[i - 1].emb_out)

        model.modules[-2].share(memory[len(memory) - 1].emb_out)

    elif share_type == 2:
        # Type 2: layer-wise weight tying
        for i in range(1, nhops):
            memory[i].emb_query.share(memory[0].emb_query)
            memory[i].emb_out.share(memory[0].emb_out)

    if add_proj:
        for i in range(1, nhops):
            proj[i].share(proj[0])

    # Cost
    loss = CrossEntropyLoss()
    loss.size_average = False
    loss.do_softmax_bprop = True
    model.modules[-1].skip_bprop = True

    return memory, model, loss