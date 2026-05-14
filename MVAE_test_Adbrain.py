# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:07:52 2019
@author: chunmanzuo
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import math
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch import optim
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from tqdm import trange
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scMVAE.utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, load_checkpoint, adjust_learning_rate
from scMVAE.MVAE_model import scMVAE_Concat, scMVAE_NN, scMVAE_POE


def train(args, adata, adata1, model, train_index, test_index, lib_mean, lib_var, lib_mean1, lib_var1, real_groups, 
          final_rate, file_fla, Type1, Type, device, scale_factor):

    train = data_utils.TensorDataset(
        torch.from_numpy(adata.raw[train_index].X.toarray()),
        torch.from_numpy(lib_mean[train_index]),
        torch.from_numpy(lib_var[train_index]),
        torch.from_numpy(lib_mean1[train_index]),
        torch.from_numpy(lib_var1[train_index]),
        torch.from_numpy(adata1.raw[train_index].X.toarray()))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    test = data_utils.TensorDataset(
        torch.from_numpy(adata.raw[test_index].X.toarray()),
        torch.from_numpy(lib_mean[test_index]),
        torch.from_numpy(lib_var[test_index]),
        torch.from_numpy(lib_mean1[test_index]),
        torch.from_numpy(lib_var1[test_index]),
        torch.from_numpy(adata1.raw[test_index].X.toarray()))
    test_loader = data_utils.DataLoader(test, batch_size=len(test_index), shuffle=False)

    total = data_utils.TensorDataset(
        torch.from_numpy(adata.raw.X.toarray()),
        torch.from_numpy(adata1.raw.X.toarray()))
    total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)

    args.max_epoch  = 500
    train_loss_list = []

    # ── Log history để tính trung bình cuối training ──────────────────────────
    ari_history  = []
    nmi_history  = []
    loss_history = []

    flag_break      = 0
    epoch_count     = 0
    reco_epoch_test = 0
    test_like_max   = 100000
    status          = ""

    args.epoch_per_test = 10

    params    = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)

    epoch     = 0
    iteration = 0
    start     = time.time()

    model.init_gmm_params(total_loader)

    # ── Header log ─────────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12} | {'ARI':>8} | {'NMI':>8} | {'Status'}")
    print("-" * 75)

    while True:

        model.train()
        epoch += 1
        epoch_lr  = adjust_learning_rate(args.lr, optimizer, epoch, final_rate, 10)
        kl_weight = min(1, epoch / args.anneal_epoch)

        # ── Train loop ─────────────────────────────────────────────────────────
        train_loss_accum = 0.0
        for batch_idx, (X1, lib_m, lib_v, lib_m1, lib_v1, X2) in enumerate(train_loader):

            X1, X2         = X1.float().to(device),  X2.float().to(device)
            lib_m, lib_v   = lib_m.to(device),        lib_v.to(device)
            lib_m1, lib_v1 = lib_m1.to(device),       lib_v1.to(device)

            X1, X2         = Variable(X1),    Variable(X2)
            lib_m, lib_v   = Variable(lib_m), Variable(lib_v)
            lib_m1, lib_v1 = Variable(lib_m1),Variable(lib_v1)

            optimizer.zero_grad()

            loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model(
                X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1)
            loss = torch.mean(
                (scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1)
                + (kl_weight * kl_divergence_z))

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            iteration += 1

        avg_train_loss = train_loss_accum / len(train_loader)
        epoch_count   += 1

        # ── Eval mỗi epoch_per_test epoch ──────────────────────────────────────
        if epoch % args.epoch_per_test == 0 and epoch > 0:

            model.eval()
            with torch.no_grad():

                # --- Test loss ---
                for batch_idx, (X1, lib_m, lib_v, lib_m1, lib_v1, X2) in enumerate(test_loader):

                    X1, X2         = X1.float().to(device),  X2.float().to(device)
                    lib_v, lib_m   = lib_v.to(device),        lib_m.to(device)
                    lib_v1, lib_m1 = lib_v1.to(device),       lib_m1.to(device)

                    X1, X2         = Variable(X1),     Variable(X2)
                    lib_m, lib_v   = Variable(lib_m),  Variable(lib_v)
                    lib_m1, lib_v1 = Variable(lib_m1), Variable(lib_v1)

                    loss1, loss2, kl_divergence_l, kl_divergence_l1, kl_divergence_z = model(
                        X1.float(), X2.float(), lib_m, lib_v, lib_m1, lib_v1)
                    test_loss = torch.mean(
                        (scale_factor * loss1 + loss2 + kl_divergence_l + kl_divergence_l1)
                        + (kl_weight * kl_divergence_z))

                train_loss_list.append(test_loss.item())

                if math.isnan(test_loss.item()):
                    flag_break = 1
                    break

                # --- ARI / NMI: lấy latent z rồi dùng nhãn GMM của model ---
                latent_z, _, _, _, _ = model.Denoise_batch(total_loader)

                if latent_z is not None:
                    # Dự đoán cụm từ latent z (dùng KMeans với n_clusters = số class)
                    n_clusters  = len(set(real_groups))
                    km          = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    pred_labels = km.fit_predict(latent_z)

                    ari = adjusted_rand_score(real_groups, pred_labels)
                    nmi = normalized_mutual_info_score(real_groups, pred_labels, average_method='arithmetic')
                else:
                    ari, nmi = 0.0, 0.0

                ari_history.append(ari)
                nmi_history.append(nmi)
                loss_history.append(test_loss.item())

                # --- Lưu model tốt nhất ---
                is_best = test_like_max > test_loss.item()
                if is_best:
                    test_like_max = test_loss.item()
                    epoch_count   = 0
                    save_checkpoint(model)

                # --- Print log mỗi eval epoch ---
                best_tag = " ✓ best" if is_best else ""
                print(f"{epoch:>6} | {avg_train_loss:>12.4f} | {test_loss.item():>12.4f} | "
                      f"{ari:>8.4f} | {nmi:>8.4f} |{best_tag}")

        # ── Điều kiện dừng (giữ nguyên logic gốc) ─────────────────────────────
        if epoch_count >= 30:
            reco_epoch_test = epoch
            status = "epoch_count > 30 (no improvement)"
            break

        if flag_break == 1:
            reco_epoch_test = epoch
            status = "NaN loss"
            break

        if epoch >= args.max_epoch:
            reco_epoch_test = epoch
            status = "reached max_epoch (500)"
            break

        if len(train_loss_list) >= 2:
            if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-4:
                reco_epoch_test = epoch
                status = "converged (loss delta < 1e-4)"
                break

    # ── Tổng kết cuối training ─────────────────────────────────────────────────
    duration = time.time() - start
    print("\n" + "=" * 75)
    print(f"  Finish training — Total time : {duration:.1f}s")
    print(f"  Stop epoch      : {reco_epoch_test}  |  Status: {status}")
    if ari_history:
        print(f"  ARI  — last: {ari_history[-1]:.4f}  |  mean: {sum(ari_history)/len(ari_history):.4f}  |  best: {max(ari_history):.4f}")
        print(f"  NMI  — last: {nmi_history[-1]:.4f}  |  mean: {sum(nmi_history)/len(nmi_history):.4f}  |  best: {max(nmi_history):.4f}")
        print(f"  Loss — last: {loss_history[-1]:.4f}  |  best: {min(loss_history):.4f}")
    print("=" * 75 + "\n")

    # ── Load lại model tốt nhất và xuất kết quả (giữ nguyên gốc) ──────────────
    load_checkpoint('./saved_model/model_best.pth.tar', model, device)

    latent_z, recon_x1, norm_x1, recon_x_2, norm_x2 = model.Denoise_batch(total_loader)

    if latent_z is not None:
        pd.DataFrame(latent_z, index=adata.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_latent_ZINB_final.csv'))
    if norm_x1 is not None:
        pd.DataFrame(norm_x1, columns=adata.var_names, index=adata.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_scRNA_norm_ZINB_final.csv'))
    if norm_x2 is not None:
        pd.DataFrame(norm_x2, columns=adata1.var_names, index=adata1.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_scATAC_norm_ZINB_final.csv'))
        
def train_with_argas( args ):

	args.workdir  =  '/content/Replication_scMVAE/scMVAE/dataset/'
	args.outdir   =  '/content/Replication_scMVAE/scMVAE/output/'

	# adata, adata1, adata2, train_index, test_index,_ = read_dataset( File1 = os.path.join( args.workdir, args.File1 ),
	# 															     File2 = os.path.join( args.workdir, args.File2 ),  
	# 															     File3 = None,
	# 															     File4 = os.path.join( args.workdir, args.File2_1 ),
	# 															     test_size_prop = 0.1
	# 															    )
	
	adata, adata1, adata2, train_index, test_index, _ = read_dataset(
		File_RNA  = os.path.join(args.workdir, 'PBMC/RNA.h5ad'),
		File_ATAC = os.path.join(args.workdir, 'PBMC/ATAC.h5ad'),
		test_size_prop = 0.1
	)

	adata  = normalize( adata,  size_factors = False, 
						normalize_input = False,  logtrans_input = True ) 

	adata1 = normalize( adata1, size_factors = False, 
						normalize_input = False, logtrans_input = True )

	print("RNA min ", adata.X.min())
	print("RNA max ", adata.X.max())
	print("RNA raw min ", adata.raw.X.min())
	print("RNA raw max ", adata.raw.X.max())
    
	print("ATAC min ", adata1.X.min())
	print("ATAC max ", adata1.X.max())
	print("ATAC raw min ", adata1.raw.X.min())
	print("ATAC raw max ", adata1.raw.X.max())

	args.batch_size     = 64
	args.epoch_per_test = 10
	
	lib_mean, lib_var   = calculate_log_library_size( adata.X )
	lib_mean1, lib_var1 = calculate_log_library_size( adata1.X )

	Nsample, Nfeature   = np.shape( adata.X )
	Nsample1, Nfeature1 = np.shape( adata1.X )

	device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
	
	model  = scMVAE_POE ( encoder_1       = [Nfeature, 1024, 128, 128],
		                  hidden_1        = 128, 
		                  Z_DIMS          = 22, 
		                  decoder_share   = [22, 128, 256],
		                  share_hidden    = 128, 
		                  decoder_1       = [128, 128, 1024], 
		                  hidden_2        = 1024, 
		                  encoder_l       = [ Nfeature, 128 ],
		                  hidden3         = 128, 
		                  encoder_2       = [Nfeature1, 1024, 128, 128], 
		                  hidden_4        = 128,
		                  encoder_l1      = [Nfeature1, 128], 
		                  hidden3_1       = 128, 
		                  decoder_2       = [128, 128, 1024],
		                  hidden_5        = 1024, 
		                  drop_rate       = 0.1, 
		                  log_variational = True,
			          Type            = "ZINB", 
			          device          = device, 
				  n_centroids     = 22, 
				  penality        = "GMM",
				  model           = 1,  )

	args.lr           = 0.001
	args.anneal_epoch = 200

	model.to(device)
	infer_data = adata1

	train( args, adata, infer_data, model, train_index, test_index, lib_mean, lib_var, 
		   lib_mean1, lib_var1, adata.obs['Group'], 0.0001, 1, "ZINB", "ZINB", device, 
		   scale_factor = 4 )


if __name__ == "__main__":

	parser = parameter_setting()
	args   = parser.parse_args()

	train_with_argas(args)
	
