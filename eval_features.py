import numpy as np
import os
import scipy
from sklearn.preprocessing import normalize

def fx_calc_map_label(view_1, view_2, label_test):
    dist = scipy.spatial.distance.cdist(view_1, view_2, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(numcases):
            if label_test[i] == label_test[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)

def eval_func(img_pairs, save_path):
    print('number of img views: ', img_pairs)
    img_feat = np.load(save_path + '/img_feat_{}.npy'.format(img_pairs))
    pt_feat = np.load(save_path + '/pt_feat.npy')
    mesh_feat = np.load(save_path + '/mesh_feat.npy')
    label = np.load(save_path + '/label.npy')
    
    img_test = normalize(img_feat, norm='l1', axis=1)
    cloud_test = normalize(pt_feat, norm='l1', axis=1)
    mesh_test = normalize(mesh_feat, norm='l1', axis=1)
    
    par_list = [
        (img_test, img_test, 'Image2Image'),
        (img_test, mesh_test, 'Image2Mesh'),
        (img_test, cloud_test, 'Image2Point'),
        (mesh_test, mesh_test, 'Mesh2Mesh'),
        (mesh_test, img_test, 'Mesh2Image'),
        (mesh_test, cloud_test, 'Mesh2Point'),
        (cloud_test, cloud_test, 'Point2Point'),
        (cloud_test, img_test, 'Point2Image'),
        (cloud_test, mesh_test, 'Point2Mesh')
    ]
    
    for index in range(9):
        view_1, view_2, name = par_list[index]
        print(name + '---------------------------')
        acc = fx_calc_map_label(view_1, view_2, label)
        acc_round = round(acc*100, 2)
        print(str(acc_round))

if __name__ == "__main__":
    save_path = 'extracted_features/ModelNet40'
    
    print("Evaluating existing features...")
    
    # Run evaluation for different numbers of image views
    eval_func(1, save_path)
    eval_func(2, save_path)
    eval_func(4, save_path)