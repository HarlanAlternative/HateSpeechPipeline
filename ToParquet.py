import os
import math
import pandas as pd
from datetime import datetime

# -------- 配置区 --------
sub_file = r"submissions/RS_2024-12_submission.json"
com_file = r"comments/RC_2024-12_comments.json"

out_dir = r"mini_dataset"
os.makedirs(out_dir, exist_ok=True)

target_subreddits = {"politics", "worldnews", "AskReddit"}

start_date = datetime(2024, 12, 1).timestamp()
end_date   = datetime(2024, 12, 7, 23, 59, 59).timestamp()

chunksize = 10**6            # 每次读取 100 万行
sub_shard_rows = 500_000     # submissions 每片行数
com_shard_rows = 1_000_000   # comments 每片行数

# 先跑小量数据：最多取多少条 post / comment；设为 None 表示不限制
MAX_SUBMISSIONS = 10000
MAX_COMMENTS    = 200000

# 可选：对 comments 再随机下采样（0~1 之间，1 表示不采样）
COMMENT_SAMPLE_FRAC = 1.0

# -------- 1) 过滤 + 分片写出 submissions --------
sub_total = 0
sub_shard_idx = 0
sub_shard_buf = []

for i, chunk in enumerate(pd.read_json(sub_file, lines=True, chunksize=chunksize)):
    mask = (
        chunk["subreddit"].isin(target_subreddits)
        & (chunk["created_utc"] >= start_date)
        & (chunk["created_utc"] <= end_date)
    )
    sub = chunk.loc[mask, ["id","author","title","selftext","subreddit","created_utc","score"]].copy()
    if not sub.empty:
        sub_shard_buf.append(sub)
        cur = pd.concat(sub_shard_buf, ignore_index=True)
        while len(cur) >= sub_shard_rows:
            part = cur.iloc[:sub_shard_rows]
            part.to_parquet(os.path.join(out_dir, f"submissions_part{sub_shard_idx}.parquet"))
            sub_total += len(part)
            sub_shard_idx += 1
            cur = cur.iloc[sub_shard_rows:]
        sub_shard_buf = [cur]
    if MAX_SUBMISSIONS is not None and sub_total + sum(len(x) for x in sub_shard_buf) >= MAX_SUBMISSIONS:
        break

# 写出最后一片
if sub_shard_buf:
    last = pd.concat(sub_shard_buf, ignore_index=True)
    if MAX_SUBMISSIONS is not None and len(last) > (MAX_SUBMISSIONS - sub_total):
        last = last.iloc[: (MAX_SUBMISSIONS - sub_total)]
    if len(last) > 0:
        last.to_parquet(os.path.join(out_dir, f"submissions_part{sub_shard_idx}.parquet"))
        sub_total += len(last)

print(f"[SUB] saved = {sub_total}")

if sub_total == 0:
    raise SystemExit("没有符合条件的 submissions，检查筛选条件或文件路径。")

# 汇总读取所有 submission 分片，提取 ID 集合
sub_id_list = []
for k in range(sub_shard_idx + 1):
    df = pd.read_parquet(os.path.join(out_dir, f"submissions_part{k}.parquet"))
    sub_id_list.append(df["id"].astype(str))
all_sub_ids = pd.concat(sub_id_list, ignore_index=True).astype(str)
sub_ids_plain = set(all_sub_ids.tolist())
sub_ids_t3    = set("t3_" + all_sub_ids)
sub_ids_union = sub_ids_plain | sub_ids_t3
print(f"[SUB] unique post ids = {len(sub_ids_union)}")

# -------- 2) 过滤 + 分片写出 comments（只保留属于这些 posts 的）--------
com_total = 0
com_shard_idx = 0
com_shard_buf = []

for i, chunk in enumerate(pd.read_json(com_file, lines=True, chunksize=chunksize)):
    mask = (
        chunk["subreddit"].isin(target_subreddits)
        & (chunk["created_utc"] >= start_date)
        & (chunk["created_utc"] <= end_date)
        & (chunk["link_id"].astype(str).isin(sub_ids_union))
    )
    com = chunk.loc[mask, ["id","link_id","parent_id","author","body","subreddit","created_utc","score"]].copy()
    if COMMENT_SAMPLE_FRAC < 1.0 and not com.empty:
        com = com.sample(frac=COMMENT_SAMPLE_FRAC, random_state=42)
    if not com.empty:
        com_shard_buf.append(com)
        cur = pd.concat(com_shard_buf, ignore_index=True)
        while len(cur) >= com_shard_rows:
            part = cur.iloc[:com_shard_rows]
            part.to_parquet(os.path.join(out_dir, f"comments_part{com_shard_idx}.parquet"))
            com_total += len(part)
            com_shard_idx += 1
            cur = cur.iloc[com_shard_rows:]
        com_shard_buf = [cur]
    if MAX_COMMENTS is not None and com_total + sum(len(x) for x in com_shard_buf) >= MAX_COMMENTS:
        break

# 写出最后一片
if com_shard_buf:
    last = pd.concat(com_shard_buf, ignore_index=True)
    if MAX_COMMENTS is not None and len(last) > (MAX_COMMENTS - com_total):
        last = last.iloc[: (MAX_COMMENTS - com_total)]
    if len(last) > 0:
        last.to_parquet(os.path.join(out_dir, f"comments_part{com_shard_idx}.parquet"))
        com_total += len(last)

print(f"[COM] saved = {com_total}")

# -------- 3) 生成汇总清单，便于后续读取 --------
sub_parts = [f for f in os.listdir(out_dir) if f.startswith("submissions_part")]
com_parts = [f for f in os.listdir(out_dir) if f.startswith("comments_part")]
pd.Series(sorted(sub_parts)).to_csv(os.path.join(out_dir, "submissions_parts.txt"), index=False, header=False)
pd.Series(sorted(com_parts)).to_csv(os.path.join(out_dir, "comments_parts.txt"), index=False, header=False)
print("[DONE] mini dataset written to:", os.path.abspath(out_dir))
