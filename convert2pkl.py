import json, re, ast, pickle, pathlib, copy

jsonl_path = pathlib.Path("post_sft_eval_result_llama3_context_w_perc_front/answers-epoch1.jsonl")   # source
pkl_path   = pathlib.Path("llama3_traj_context_w_perc_front-epoch1.pkl")        # destination
log_path   = pathlib.Path("malformed_tokens.txt")
token2traj   = {}
bad_tokens   = []          # collect tokens that couldn’t be parsed
traj_pat     = re.compile(r"3[- ]second trajectory:\s*(\[[^\]]+\])", re.I)
DEFAULT_TRAJ = [[0.0, 0.0]] * 6                     # 6 waypoints at origin

with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        obj   = json.loads(line)
        token = obj["question_id"]
        text  = obj["text"]
        m = traj_pat.search(text)
        if not m:
            # no trajectory string at all
           
            traj = copy.deepcopy(DEFAULT_TRAJ)
            bad_tokens.append(token)
        else:
            try:
                traj = [list(pt) for pt in ast.literal_eval(m.group(1))]
                if len(traj) < 6:              # pad
                    traj.extend([traj[-1]] * (6 - len(traj)))
                elif len(traj) > 6:            # trim
                    traj = traj[:6]
            except (ValueError, SyntaxError, TypeError):
                # malformed list → fallback
                traj = copy.deepcopy(DEFAULT_TRAJ)
                bad_tokens.append(token)

        token2traj[token] = traj
        

# save pickle
with pkl_path.open("wb") as f:
    pickle.dump(token2traj, f)

# log malformed tokens for review
if bad_tokens:
    with log_path.open("w") as f:
        f.write("\n".join(bad_tokens))
    print(f"{len(bad_tokens)} malformed trajectory entries written to {log_path}")

print(f"Processed {len(token2traj):,} records ➜ {pkl_path}")