"""Evaluate fixed answer selection rules on new paired short natural scenes."""
import fcntl
import os

import numpy as np
import torch

from ilm.visual_lm.pictured_answer_selection import AnswerSelection
from scripts.train_pictured_answer_selection import features, store_streams, ROOT, WORK, PUBLIC, READER, FontInvariantReader, record, sha
from scripts.evaluate_pointing_causal_writer import generate, PointingCausalWriter
from scripts.fit_font_invariant_reader import Face, FONTS
from scripts.pointing_glyph_corpus import Row, digest

# Each pair needs a different name. Repeat after exchanging names in the source.
SCENES = (
    ("廚房水管漏了，{b}去找工具，{a}留在原地扶住水管。","誰去拿工具？","誰在扶水管？"),
    ("{b}從衣袋掏出一枚舊幣，{a}拿放大鏡察看幣上的字。","誰帶來了舊幣？","誰仔細看幣上的字？"),
    ("遊戲開始前，{b}把眼睛蒙上，{a}把鈴鐺藏進木盒。","誰看不見藏鈴鐺的過程？","誰決定把鈴鐺放在哪裏？"),
    ("船靠岸時是{b}拋出繩子，岸上的{a}把繩子繫在樹上。","誰把船繩扔向岸邊？","誰最後繫緊了船繩？"),
    ("展覽那天，{b}把畫掛上牆，{a}則逐一填寫作品的標籤。","誰負責掛畫？","標籤上的字是誰寫的？"),
    ("停電以後，{b}點起一盞燈，{a}借着燈光把地址抄了下來。","誰提供了光源？","誰抄寫了地址？"),
    ("{b}拿着空瓶去井邊打水，{a}留在營地看守行李。","誰離開了營地？","誰留守行李？"),
    ("收據上寫的是{b}的名字，替商店在右下角蓋章的卻是{a}。","收據開給誰？","誰代表商店蓋章？"),
    ("{b}敲了三下門，屋內的{a}聽見聲響才放下書。","誰站在門外？","誰剛才在讀書？"),
    ("開庭時{b}坐在證人席，{a}把證詞逐句記在紙上。","誰出庭作證？","誰記下了證詞？"),
    ("幻燈片由{b}事先做完，當天站在臺上解說的是{a}。","誰製作幻燈片？","誰在現場解說？"),
    ("山腳下，{b}把望遠鏡借給{a}，自己只好用肉眼看遠處。","望遠鏡原來是誰的？","誰用上了望遠鏡？"),
    ("{b}整理好一疊照片，請{a}替每張照片寫上拍攝地點。","誰先整理了照片？","誰負責補寫地點？"),
    ("哨音一響，守門的{b}向右一撲，{a}踢出的球卻飛向左邊。","誰是守門員？","誰射了門？"),
    ("{b}先在紙上摺出一隻鶴，{a}照着樣子摺了第二隻。","誰先示範了摺法？","誰照樣子學摺？"),
    ("{b}提議把牆刷白，{a}則想保留磚牆原來的紅色。","誰希望改變牆的顏色？","誰希望保留原色？"),
)
NAMES=(("周映","林嶼"),("梁曦","沈蓉"),("汪翰","程晴"),("田璐","許晗"))


def main():
    face=Face((FONTS[0],2))
    rows=[]
    for i,(p,q,t) in enumerate(SCENES):
        a,b=NAMES[i%len(NAMES)]
        for u,v in ((a,b),(b,a)):
            for question,answer in ((q,v),(t,u)):
                rows.append(Row("transfer",str(len(rows)),face(p.format(a=u,b=v)),face(question),face(answer),b""))
    x,quotes=features(FontInvariantReader.restore(READER,"cuda"),rows)
    streams,ends=generate(PointingCausalWriter.restore(WORK/"pointing","cuda"),rows)
    with torch.no_grad():
        choices={k:AnswerSelection.restore(WORK/(k+"_selection.safetensors"))(x).flatten().ge(0).tolist()
                 for k in ("learned","shuffled","sentences","random_windows")}
    choices.update(quote=[True]*len(rows),write=[False]*len(rows))
    result={}
    for k,flags in choices.items():
        output=[a if f else b for a,b,f in zip(quotes,streams,flags)]
        stopped=[s or f for s,f in zip(ends,flags)]
        ok=np.array([r.y==a and s for r,a,s in zip(rows,output,stopped)])
        file=WORK/("transfer_"+k+".npz")
        store_streams(file,output,stopped)
        result[k]=dict(exact=int(ok.sum()),n=len(rows),pairs=int(ok.reshape(-1,2).all(1).sum()),scenes=int(ok.reshape(-1,4).all(1).sum()),
                       quoted=sum(flags),stream_sha256=sha(file))
    record(PUBLIC/"transfer.json",dict(rows_sha256=digest(rows),scores=result))
    print(result,flush=True)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
