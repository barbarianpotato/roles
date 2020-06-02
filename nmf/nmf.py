from nmf import mdl
import numpy as np
import nimfa


def nmf (nodeFeatureMatrix):
    actual_fx_matrix = nodeFeatureMatrix
    n, f = actual_fx_matrix.shape
    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    best_W = None
    best_H = None
    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0
    for rank in range(1, max_roles+1):
        lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=100)
        lsnmf_fit = lsnmf()
        W = np.asarray(lsnmf_fit.basis())
        H = np.asarray(lsnmf_fit.coef())
        estimated_matrix = np.asarray(np.dot(W, H))

        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        description_length = model_cost - loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_W = np.copy(W)
            best_H = np.copy(H)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 4:
                break

        # print ('Number of Roles: %s, Model Cost: %.2f, -loglikelihood: %.2f, Description Length: %.2f, MDL: %.2f (%s)' % (rank, model_cost, loglikelihood, description_length, minimum_description_length, best_W.shape[1]))

    # print ('MDL has not changed for these many iters:', min_des_not_changed_counter)
    print ('MDL: %.2f, Roles: %s' % (minimum_description_length, best_W.shape[1]))

    return (best_W , best_H)

