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

from scMVAE.utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, load_checkpoint, adjust_learning_rate, set_seed, seed_worker, make_generator
from scMVAE.MVAE_model import scMVAE_Concat, scMVAE_NN, scMVAE_POE
import scMVAE.MVAE_model as MVAE_model_module
from scMVAE.export import save_embedding


def export_npz(args, adata, latent_z, pred_labels, stop_epoch):
    ### ghi {method}_{dataset}.npz theo contract trong EXPORT_NPZ_GUIDE.md de ve UMAP
    npz_dir = getattr(args, 'npz_dir', None) or args.outdir
    os.makedirs(npz_dir, exist_ok=True)

    # chay nhieu seed -> moi seed 1 file rieng, tranh ghi de
    seeds    = getattr(args, 'seeds', [args.seed])
    suffix   = f"_seed{args.seed}" if len(seeds) > 1 else ""
    out_path = os.path.join(npz_dir, f"{args.method_name}_{args.dataset_name}{suffix}.npz")

    # total_loader shuffle=False -> hang i cua latent_z / pred_labels ung voi adata.obs_names[i]
    y_true = adata.obs['Group'].astype(str).to_numpy()
    y_pred = np.asarray(pred_labels).astype(int)

    save_embedding(
        out_path   = out_path,
        emb        = latent_z,
        cell_ids   = adata.obs_names.to_numpy().astype(str),
        y_true     = y_true,
        y_pred     = y_pred,
        method     = args.method_name,
        ari        = adjusted_rand_score(y_true, y_pred),
        nmi        = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'),
        emb_source = "latent_z (PoE joint posterior mean, model.eval)",
        checkpoint = "best_test_loss",
        seed       = args.seed,
        stop_epoch = stop_epoch,
    )

    return out_path


def train(args, adata, adata1, model, train_index, test_index, lib_mean, lib_var, lib_mean1, lib_var1, real_groups, 
          final_rate, file_fla, Type1, Type, device, scale_factor):


    train = data_utils.TensorDataset(
        torch.from_numpy(adata.raw[train_index].X.toarray()),
        torch.from_numpy(lib_mean[train_index]),
        torch.from_numpy(lib_var[train_index]),
        torch.from_numpy(lib_mean1[train_index]),
        torch.from_numpy(lib_var1[train_index]),
        torch.from_numpy(adata1.raw[train_index].X.toarray()))
    # generator + worker_init_fn: thu tu shuffle cua batch lap lai duoc giua cac lan chay
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                         generator=make_generator(args.seed),
                                         worker_init_fn=seed_worker)

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

    # checkpoint rieng cho tung seed -> tranh seed nay load nham checkpoint cua seed truoc
    ckpt_path       = os.path.join('./saved_model', f'model_best_seed{args.seed}.pth.tar')
    saved_any       = False

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
                    km          = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
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
                    save_checkpoint(model, ckpt_path)
                    saved_any     = True

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
    if saved_any:
        load_checkpoint(ckpt_path, model, device)
    else:
        print("  [Warning] Khong co checkpoint nao duoc luu (NaN ngay tu lan eval dau) "
              "-> dung trong so hien tai.")

    model.eval()
    latent_z, recon_x1, norm_x1, recon_x_2, norm_x2 = model.Denoise_batch(total_loader)

    # ── Metric cuối cùng: clustering trên latent của best checkpoint ───────────
    final_ari, final_nmi = 0.0, 0.0
    npz_path             = None
    if latent_z is not None:
        n_clusters  = len(set(real_groups))
        km          = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
        pred_labels = km.fit_predict(latent_z)
        final_ari   = adjusted_rand_score(real_groups, pred_labels)
        final_nmi   = normalized_mutual_info_score(real_groups, pred_labels, average_method='arithmetic')
        print(f"  [Best checkpoint] ARI: {final_ari:.4f}  |  NMI: {final_nmi:.4f}\n")

        # ── Xuat .npz de ve UMAP: dung dung latent_z + pred_labels vua dung de bao cao ──
        if getattr(args, 'export_npz', False):
            npz_path = export_npz(args, adata, latent_z, pred_labels, reco_epoch_test)

    if latent_z is not None:
        pd.DataFrame(latent_z, index=adata.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_latent_ZINB_final.csv'))
    if norm_x1 is not None:
        pd.DataFrame(norm_x1, columns=adata.var_names, index=adata.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_scRNA_norm_ZINB_final.csv'))
    if norm_x2 is not None:
        pd.DataFrame(norm_x2, columns=adata1.var_names, index=adata1.obs_names).to_csv(
            os.path.join(args.outdir, str(file_fla) + '_scATAC_norm_ZINB_final.csv'))

    return {
        'seed'      : args.seed,
        'ari'       : final_ari,                # metric cua best checkpoint  <- dung de bao cao
        'nmi'       : final_nmi,
        'ari_best'  : max(ari_history) if ari_history else 0.0,
        'nmi_best'  : max(nmi_history) if nmi_history else 0.0,
        'ari_last'  : ari_history[-1] if ari_history else 0.0,
        'nmi_last'  : nmi_history[-1] if nmi_history else 0.0,
        'stop_epoch': reco_epoch_test,
        'status'    : status,
        'test_loss' : test_like_max,
        'npz_path'  : npz_path,
    }

def train_with_argas( args ):

	args.workdir  =  '/content/Replication_scMVAE/scMVAE/dataset/'
	args.outdir   =  '/content/Replication_scMVAE/scMVAE/output/'

	os.makedirs( args.outdir, exist_ok = True )

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
     
	# common = adata.obs_names.intersection(adata1.obs_names)
	# if len(common) < max(adata.n_obs, adata1.n_obs):
	# 	print(f"[Warning] Sau normalize: RNA={adata.n_obs} | ATAC={adata1.n_obs} → còn {len(common)} cells")

	# 	# O(1) lookup
	# 	rna_name_to_idx = {name: i for i, name in enumerate(adata.obs_names)}

	# 	rna_idx     = np.array([rna_name_to_idx[c] for c in common])
	# 	rna_idx_set = set(rna_idx.tolist())
	# 	old_to_new  = {old: new for new, old in enumerate(rna_idx)}

	# 	adata  = adata[common].copy()
	# 	adata1 = adata1[common].copy()

	# 	# Remap train/test index
	# 	train_index = np.array([old_to_new[i] for i in train_index if i in rna_idx_set])
	# 	test_index  = np.array([old_to_new[i] for i in test_index  if i in rna_idx_set])

	# print(f"Train: {len(train_index)} | Test: {len(test_index)} | Total: {adata.n_obs}")
	# print(f"adata.raw shape : {adata.raw.X.shape}")
	# print(f"adata1.raw shape: {adata1.raw.X.shape}")
          
	# print("RNA min ", adata.X.min())
	# print("RNA max ", adata.X.max())
	# print("RNA raw min ", adata.raw.X.min())
	# print("RNA raw max ", adata.raw.X.max())
    
	# print("ATAC min ", adata1.X.min())
	# print("ATAC max ", adata1.X.max())
	# print("ATAC raw min ", adata1.raw.X.min())
	# print("ATAC raw max ", adata1.raw.X.max())

	# print(adata.shape)

	# print(adata.X.shape)
     
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

	# file_fla = seed  ->  moi seed xuat CSV rieng, khong ghi de len nhau
	return train( args, adata, infer_data, model, train_index, test_index, lib_mean, lib_var,
			      lib_mean1, lib_var1, adata.obs['Group'], 0.0001, args.seed, "ZINB", "ZINB", device,
			      scale_factor = 4 )


def run_multi_seed( args, seeds ):
	### chay lai toan bo pipeline cho tung seed roi tong hop mean +/- std

	results = []

	for i, seed in enumerate(seeds):
		print("\n" + "#" * 75)
		print(f"#  RUN seed = {seed}   ({i + 1}/{len(seeds)})")
		print("#" * 75)

		args.seed = seed
		set_seed( seed )                      # phai goi TRUOC read_dataset / khoi tao model

		# GMM init cung phai doi theo seed, neu khong se bo sot mot nguon bien thien
		MVAE_model_module.GMM_SEED = seed

		results.append( train_with_argas(args) )

	# ── Bang ket qua tung seed ────────────────────────────────────────────────
	print("\n" + "=" * 75)
	print("  KET QUA TONG HOP")
	print("=" * 75)
	print(f"{'Seed':>6} | {'ARI':>8} | {'NMI':>8} | {'Epoch':>6} | Status")
	print("-" * 75)
	for r in results:
		print(f"{r['seed']:>6} | {r['ari']:>8.4f} | {r['nmi']:>8.4f} | {r['stop_epoch']:>6} | {r['status']}")

	ari = np.array([r['ari'] for r in results])
	nmi = np.array([r['nmi'] for r in results])

	print("-" * 75)
	print(f"  ARI : {ari.mean():.4f} +/- {ari.std(ddof=1):.4f}   (min {ari.min():.4f} / max {ari.max():.4f})")
	print(f"  NMI : {nmi.mean():.4f} +/- {nmi.std(ddof=1):.4f}   (min {nmi.min():.4f} / max {nmi.max():.4f})")
	print("=" * 75 + "\n")

	df = pd.DataFrame(results)
	out_csv = os.path.join(args.outdir, 'multi_seed_results.csv')
	df.to_csv(out_csv, index=False)
	print(f"  Da luu chi tiet -> {out_csv}")

	return df


if __name__ == "__main__":

	parser = parameter_setting()
	parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
						help='Danh sach seed de chay lap lai (vd: --seeds 0 1 2 3 4)')
	# ── Args xuat .npz (khong anh huong train / gom cum) ──────────────────────
	parser.add_argument('--no_export_npz', dest='export_npz', action='store_false',
						help='Tat xuat file .npz de ve UMAP (mac dinh: bat)')
	parser.add_argument('--npz_dir', type=str, default=None,
						help='Thu muc luu .npz (mac dinh: args.outdir)')
	parser.add_argument('--method_name', type=str, default='scMVAE',
						help='Ten method hien thi tren hinh UMAP va trong ten file .npz')
	parser.add_argument('--dataset_name', type=str, default='PBMC',
						help='Ten dataset trong ten file .npz')
	args   = parser.parse_args()

	run_multi_seed( args, args.seeds )
	
