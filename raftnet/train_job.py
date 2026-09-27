# Part of RAFT-Net. The notebooks write this file into the models/ folder of a ForMa clone
# (https://github.com/multimediaFor/ForMa) and import it from there.

# train_job.py -- trains ONE (variant, seed, acr) job for common.EPOCHS epochs, resumable across sessions.
import os, sys, json, time, argparse, shutil
ap = argparse.ArgumentParser()
for a in ('--job', '--variant', '--models_dir', '--repo_dir', '--run_dir', '--imagenet_ckpt', '--split_json'):
    ap.add_argument(a, required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--acr', type=int, required=True)
ap.add_argument('--max_hours', type=float, required=True)
args = ap.parse_args()
sys.path.insert(0, args.models_dir)
sys.path.insert(0, args.run_dir)
T0 = time.time()
J = args.job
CK = os.path.join(args.run_dir, 'ckpt')
os.makedirs(CK, exist_ok=True)


def log(*a):
    print(f"[{J}]", *a, flush=True)


def write_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def status(state, **kw):
    write_json(dict(state=state, job=J, variant=args.variant, seed=args.seed, acr=bool(args.acr), **kw),
               os.path.join(CK, f'{J}_status.json'))


def find_state(job, bases=('/kaggle/working', '/kaggle/input')):
    """Most advanced saved state for this job, from this session or an attached previous version's output."""
    best, best_ep = None, -1
    for base in bases:
        for r, _, fs in os.walk(base):
            if f'{job}_state.json' in fs and f'{job}_state.pth' in fs:
                try:
                    ep = json.load(open(os.path.join(r, f'{job}_state.json')))['epoch']
                except Exception:
                    continue
                if ep > best_ep:
                    best, best_ep = r, ep
    return best, best_ep


if __name__ == '__main__':
    import numpy as np
    import torch
    import torch.optim as optim
    import common as C
    from raftnet import build_model, load_imagenet_backbone
    from metrics_engine import MetricAccumulator

    assert not (args.acr and args.variant.startswith('forma')), 'ACR needs return_feat (RAFT-Net variants only)'
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    split = json.load(open(args.split_json))
    TRAIN = [tuple(x) for x in split['train']]
    VAL = [tuple(x) for x in split['val']]
    assert (len(TRAIN), len(VAL)) == (11225, 1247)
    forma_loss = C.init_forma_loss(args.repo_dir)
    NP = os.path.join(args.repo_dir, 'weights', 'np++.pth')

    C.set_seed(args.seed)
    model = build_model(args.variant, NP).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=C.LR, weight_decay=C.WD)
    steps = len(TRAIN) // C.BATCH
    sched = C.make_schedule(opt, steps)
    scaler = torch.amp.GradScaler('cuda')
    start, best, hist = 0, -1.0, []
    meta = dict(job=J, variant=args.variant, seed=args.seed, acr=bool(args.acr), epochs=C.EPOCHS, batch=C.BATCH,
                lr=C.LR, weight_decay=C.WD, input=C.INPUT, acr_start_epoch=C.ACR_START_EPOCH, acr_lambda=C.ACR_LAMBDA,
                params_m=sum(p.numel() for p in model.parameters()) / 1e6, color='rgb', normalize='imagenet')

    src_dir, src_ep = find_state(J)
    if src_dir is not None:
        st = torch.load(os.path.join(src_dir, f'{J}_state.pth'), map_location='cpu', weights_only=False)
        assert st['meta']['variant'] == args.variant and st['meta']['seed'] == args.seed and st['meta']['acr'] == bool(args.acr)
        model.load_state_dict(st['model']); opt.load_state_dict(st['optimizer'])
        sched.load_state_dict(st['scheduler']); scaler.load_state_dict(st['scaler'])
        start, best, hist = st['epoch'], st['best_f1'], st['history']
        if os.path.abspath(src_dir) != os.path.abspath(CK):          # carry best-so-far files into this session's output
            for f in (f'{J}_best.pth', f'{J}_meta.json', f'{J}_history.json'):
                if os.path.exists(os.path.join(src_dir, f)):
                    shutil.copy(os.path.join(src_dir, f), os.path.join(CK, f))
        log(f"RESUMED at epoch {start + 1}/{C.EPOCHS} from {src_dir} | best val F1 so far {best:.4f}")
        del st
    else:
        raw = torch.load(args.imagenet_ckpt, map_location='cpu', weights_only=False)
        n_loaded, n_ck, _ = load_imagenet_backbone(model, raw['model'] if 'model' in raw else raw)
        del raw
        log(f"fresh run | ImageNet tensors loaded {n_loaded}/{n_ck} | {steps} steps/epoch | params {meta['params_m']:.2f} M")
        # step-0 self-check on a real batch (no optimizer step): finite losses, gradients present
        x, m = next(iter(C.train_loader(TRAIN, args.seed, 0)))
        x, m = x.to(dev), m.to(dev)
        loss, parts = C.main_loss(model, x, m, forma_loss)
        loss.backward()
        assert torch.isfinite(loss), 'non-finite loss at step 0'
        if args.acr:
            a = C.acr_subbatch_loss(model, x)
            a.backward()
            assert torch.isfinite(a), 'non-finite ACR at step 0'
        assert sum(int(p.grad is not None) for p in params) > 0.9 * len(params), 'gradients missing'
        opt.zero_grad(set_to_none=True)
        log(f"self-check passed: loss {float(loss):.4f} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()))
        del x, m, loss

    if start >= C.EPOCHS:
        status('done', epoch=start, best_f1=best)
        log('already complete')
        sys.exit(0)
    status('running', epoch=start, best_f1=best)
    vloader = C.val_loader(VAL)
    epoch_h, bad_epochs = [], 0
    for epoch in range(start, C.EPOCHS):
        if epoch_h and (time.time() - T0) / 3600 + 1.15 * max(epoch_h[-3:]) > args.max_hours:
            status('paused', epoch=epoch, best_f1=best)
            log(f"PAUSED before epoch {epoch + 1} (session time). Re-run the notebook with this version's output attached.")
            sys.exit(0)
        lam = C.acr_lambda(epoch) if args.acr else 0.0
        t0, skipped, n, sums, acr_sum = time.time(), 0, 0, {}, 0.0
        w0 = C.param_sq(model)
        model.train()
        for x, m in C.train_loader(TRAIN, args.seed, epoch):
            x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, parts = C.main_loss(model, x, m, forma_loss)
            scaler.scale(loss).backward()                        # main graph freed before the ACR pass
            if lam > 0:
                a = C.acr_subbatch_loss(model, x)
                scaler.scale(lam * a).backward()
                acr_sum += float(a.detach())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            s0 = scaler.get_scale()
            scaler.step(opt); scaler.update(); sched.step()
            skipped += int(scaler.get_scale() < s0)
            n += 1
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
        v = C.validate(model, vloader, dev, MetricAccumulator)
        epoch_h.append((time.time() - t0) / 3600)
        rec = dict(epoch=epoch + 1, hours=epoch_h[-1], lr=opt.param_groups[0]['lr'], acr_lambda=lam,
                   acr=acr_sum / max(n, 1), skipped=skipped, weight_change=abs(C.param_sq(model) - w0),
                   val_f1=v['img_mean_f1'], val_iou=v['img_mean_iou'], val_auc=v.get('pixel_roc_auc', float('nan')),
                   **{k: s / max(n, 1) for k, s in sums.items()})
        hist.append(rec)
        improved = rec['val_f1'] > best
        if improved:
            best = rec['val_f1']
            torch.save(model.state_dict(), os.path.join(CK, f'{J}_best.pth.tmp'))
            os.replace(os.path.join(CK, f'{J}_best.pth.tmp'), os.path.join(CK, f'{J}_best.pth'))
            write_json({**meta, 'best_epoch': epoch + 1, 'best_val': rec}, os.path.join(CK, f'{J}_meta.json'))
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best_f1': best, 'history': hist, 'meta': meta},
                   os.path.join(CK, f'{J}_state.pth.tmp'))
        os.replace(os.path.join(CK, f'{J}_state.pth.tmp'), os.path.join(CK, f'{J}_state.pth'))
        write_json({'epoch': epoch + 1, 'best_f1': best}, os.path.join(CK, f'{J}_state.json'))
        write_json(hist, os.path.join(CK, f'{J}_history.json'))
        status('running', epoch=epoch + 1, best_f1=best)
        log(f"epoch {epoch + 1}/{C.EPOCHS} | " + " ".join(f"{k} {rec[k]:.4f}" for k in sums) +
            (f" | acr {rec['acr']:.4f} (lambda {lam:.4f})" if lam > 0 else "") +
            f" | val F1 {rec['val_f1']:.4f} IoU {rec['val_iou']:.4f} AUC {rec['val_auc']:.4f} | best {best:.4f}"
            f"{' *' if improved else ''} | skipped {skipped}/{n} | lr {rec['lr']:.2e} | {60 * epoch_h[-1]:.1f} min")
        bad_epochs = bad_epochs + 1 if (skipped > 0.5 * n or rec['weight_change'] == 0.0) else 0
        if bad_epochs >= 2:
            status('failed', epoch=epoch + 1, best_f1=best, reason='frozen: most steps skipped or weights not moving')
            log("STOPPED: two consecutive epochs with most optimizer steps skipped or no weight movement. Report this log.")
            sys.exit(3)
    status('done', epoch=C.EPOCHS, best_f1=best)
    log(f"DONE | best val F1 {best:.4f} (epoch {max(hist, key=lambda h: h['val_f1'])['epoch']})")
