def umap(features, d=3):
    import umap
    model = umap.UMAP(n_components=d)
    embedding = model.fit_transform(features)
    return embedding

def tsne(features, d=3):
    from sklearn.manifold import TSNE
    model = TSNE(n_components=d)
    embedding = model.fit_transform(features)
    return embedding

def pacmap(features,d=3):
    import pacmap
    model = pacmap.PaCMAP(n_components=d)
    embedding = model.fit_transform(features)
    return embedding

def pca(features, d=3):
    from sklearn.decomposition import PCA
    model = PCA(n_components=d)
    embedding = model.fit_transform(features)
    return embedding

projectors = {
    'umap': umap,
    'tsne': tsne,
    'pacmap': pacmap,
    'pca': pca
}