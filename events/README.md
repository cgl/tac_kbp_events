# TAC KBP EVENTS VISUALIZE

Too see event mentions on given dataset, first you need to download LDC2017E02 dataset on place it under `data` folder. Then you can see event mentions highlighted using below command:

    python visualization/visualize_on_console.py


### Toy Examples

    from prepare_datafile import load_vocab, load_bin_vec, initialize ; initialize() ; vocab = load_vocab()
    w2v_file= "/datasets//GoogleNews-vectors-negative300.bin"
    vocab = set([word.lower() for word in vocab if not word.isalnum()])
    dim, word_vecs = load_bin_vec(w2v_file, vocab)



    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format('/datasets/GoogleNews-vectors-negative300.bin', binary=True)
    model.save_word2vec_format('/datasets/GoogleNews-vectors-negative300.txt', binary=False)
