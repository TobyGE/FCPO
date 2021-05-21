import numpy as np
# import torch 

def get_test_results(test_res, user_history_length, gini_index, max_k):
    def get_ranking_evaluation(rec, len_history, max_k):
        '''
        @input:
        - rec: [reindexed iid], the recommendation list
        - rec_scores: corresponding scores for rec
        - hist: [reindexed iid], the user history
        - hist_scores: corresponding scores for hist
        - max_k: maximum observation length
        '''
        
        tp = np.zeros(max_k)
        dcg = np.zeros(max_k)
        idcg = np.zeros(max_k)
        if rec[0] == 1:
            tp[0] = 1
            dcg[0] = 1
        idcg[0] = 1
        for i in range(1,max_k):
            if rec[i] == 1:
                tp[i] = tp[i-1] + 1
                dcg[i] = dcg[i-1] + 1.0 / np.log2(i+2)
            else:
                tp[i] = tp[i-1]
                dcg[i] = dcg[i-1]
            if i < len_history:
                idcg[i] = idcg[i-1] + 1.0 / np.log2(i+2)
            else:
                idcg[i] = idcg[i-1]


        # ndcg = dcg / idcg
        ndcg = dcg / idcg
        # recall = TP / (TP + FN)
        if len_history == 0:
            recall = 0
        else:
            recall = tp / len_history
        # precision = TP / (TP + FP)
        precision = tp / np.arange(1,max_k + 1)
        # hit_rate = 1 if the list till k has a hit in the hist
        hit_rate = tp
        hit_rate[tp > 0] = 1

        return {"ndcg": ndcg, "recall": recall, "precision": precision, "hit_rate": hit_rate}
    
    N = test_res.shape[0]
    recall = np.zeros((N, max_k))
    hit_rate = np.zeros((N, max_k))
    precision = np.zeros((N, max_k))
    ndcg = np.zeros((N, max_k))
    for i in range(N):
        report = get_ranking_evaluation(test_res[i], user_history_length[i], max_k)
        recall[i] = report["recall"]
        hit_rate[i] = report["hit_rate"]
        precision[i] = report["precision"]
        ndcg[i] = report["ndcg"]
        
    return {"recall": recall, "hit_rate": hit_rate, "precision": precision, "ndcg": ndcg, "gini_index": gini_index}