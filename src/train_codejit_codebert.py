    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def encode_all(messages, patches):
        ids, flags = [], []
        for m, c in zip(messages, patches):
            t, mt, ct = build_tokens(tokenizer, m, c, args.input_mode, args.max_length, args.message_budget)
            if len(t) > args.max_length:
                raise AssertionError('Token limit exceeded')
            ids.append(t)
            flags.append((mt, ct, len(t)))
        return ids, flags
    tr_ids, tr_flags = encode_all(train_msg, train_patch)
    va_ids, va_flags = encode_all(val_msg, val_patch)
    class Rows(Dataset):
        def __init__(self, ids, labels):
            self.ids = ids; self.labels = labels
        def __len__(self): return len(self.ids)
        def __getitem__(self, i): return self.ids[i], int(self.labels[i])
    def collate(rows):
        sequences, labels = zip(*rows)
        enc = tokenizer.pad({'input_ids': list(sequences)}, padding=True, return_tensors='pt')
        enc['labels'] = torch.tensor(labels, dtype=torch.long)
        return enc
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Rows(tr_ids, ytrain), batch_size=args.batch_size, shuffle=True,
                              generator=gen, collate_fn=collate)
    val_loader = DataLoader(Rows(va_ids, yval), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).cuda()
    class_counts = np.bincount(ytrain, minlength=2)
    weights = torch.tensor(len(ytrain) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32, device='cuda')
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    epochs = 1 if args.smoke_test else args.epochs
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(.1 * steps_per_epoch * epochs),
                                                 num_training_steps=steps_per_epoch * epochs)
    scaler = torch.amp.GradScaler('cuda')
    @torch.no_grad()
    def predict():
        model.eval(); probs = []
        for batch in val_loader:
            inp = {k: v.cuda() for k, v in batch.items() if k != 'labels'}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**inp).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return np.asarray(probs)
    history = []; best = -1.; stale = 0; best_threshold = .5
    best_path = out / 'best_model'
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); losses = []
        for idx, batch in enumerate(train_loader):
            group_start = (idx // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                raw_loss = loss_fn(logits.float(), batch['labels'])
            scaler.scale(raw_loss / group_size).backward()
            losses.append(float(raw_loss.item()))
            if (idx + 1) % args.grad_accum == 0 or idx + 1 == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                before = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
        probs = predict()
        threshold = tune_threshold(yval, probs)
        scores = metric(yval, probs, threshold)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), tuned_threshold=threshold,
                            validation_tuned=scores, validation_default_05=metric(yval, probs, .5)))
        print(f'{args.input_mode} epoch {epoch}: loss={np.mean(losses):.4f} '
              f'val F1@0.5={history[-1]["validation_default_05"]["f1"]:.4f} '
              f'val tuned F1={scores["f1"]:.4f} val AP={scores["pr_auc"]:.4f}', flush=True)
        # Selection criterion: tuned validation F1, exploratory, no test involved.
        if scores['f1'] > best:
            best = scores['f1']; stale = 0; best_threshold = threshold
            model.save_pretrained(best_path); tokenizer.save_pretrained(best_path)
        else:
            stale += 1
            if stale >= args.patience: break
    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(best_path).cuda()
    probs = predict()
    pred = (probs >= .5).astype(int)
    pd.DataFrame(dict(model=args.input_mode, repo=val.repo.to_numpy(), commit_id=val.commit_id.to_numpy(),
                      label=yval, predicted=pred, probability_positive=probs,
                      predicted_tuned=(probs >= best_threshold).astype(int))).to_csv(
                          out / 'validation_predictions.csv', index=False)
    coverage_rows = []
    for subset, frame, flags in [('train', train, tr_flags), ('val', val, va_flags)]:
        for r, (mt, ct, length) in zip(frame.itertuples(index=False), flags):
            coverage_rows.append(dict(subset=subset, repo=r.repo, commit_id=r.commit_id,
                                      message_tokens_truncated=mt, code_tokens_truncated=ct,
                                      model_tokens=length))
    pd.DataFrame(coverage_rows).to_csv(out / 'token_coverage.csv', index=False)
    result = dict(experiment='CodeJIT CodeBERT repository-disjoint validation-only',
                  input_mode=args.input_mode, model_name=args.model_name,
                  data_audit=data_report, selected_epoch=int(history[int(np.argmax([x['validation_tuned']['f1'] for x in history]))]['epoch']),
                  threshold_tuned_on_validation=best_threshold,
                  validation_default_05=metric(yval, probs, .5),
                  validation_tuned=metric(yval, probs, best_threshold),
                  training_history=history, seconds=float(time.time()-started),
                  config=vars(args),
                  limitations=['Curated CodeJIT negatives do not represent ordinary CI commits.',
                               'No chronological split; repositories can share code/forks.',
                               'Patch prefix and tokenizer truncation may omit evidence.',
                               'Validation-based checkpoint/threshold selection is exploratory.',
                               'Held-out test cohort never accessed.'])
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mode=args.input_mode, selected_epoch=result['selected_epoch'],
                          validation_default_05=result['validation_default_05'],
                          validation_tuned=result['validation_tuned'], output=str(out)), indent=2))

if __name__ == '__main__':
    main()
