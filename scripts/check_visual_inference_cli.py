"""Verify normal typed and PNG inference on a small fixed set of newly authored Chinese cases."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

from scripts import ask_visual_inference as cli
from scripts.train_visual_inference_writer import PUBLIC, WORK, Rasterizer, load,sha,write_json
from scripts.prepare_visual_inference_pairs import RESPONSES

EXAMPLES=(
    ("restore","她端起杯子，喝了一口水。","她喝了一口□。","水"),
    ("restore","她端起杯子，喝了一口茶。","她喝了一口□。","茶"),
    ("restore","小陈关上门，离开了教室。","小陈关上□。","门"),
    ("restore","小陈关上窗，离开了教室。","小陈关上□。","窗"),
    ("decide","盒中放着三本书。","盒子里至少有一本书。",RESPONSES[0]),
    ("decide","盒子是空的。","盒子里至少有一本书。",RESPONSES[1]),
)
REG=PUBLIC/"cli_cases.json"


def run():
    binding=load(REG)
    assert binding["source_sha256"]==sha(Path(__file__)) and binding["cli_sha256"]==sha(Path(cli.__file__))
    out=WORK/"cli"
    out.mkdir()
    r=Rasterizer();results=[]
    for index,(mode,p,h,y) in enumerate(EXAMPLES):
        typed=out/(str(index)+"_typed");png=out/(str(index)+"_image")
        for path,flags in ((typed,["--premise-text",p,"--claim-text",h]),(png,["--premise-image",str(typed/"premise.png"),"--claim-image",str(typed/"claim.png")])):
            sys.argv=["ask_visual_inference","--mode",mode,*flags,"--output",str(path)]
            with redirect_stdout(StringIO()):cli.main()
        receipt=load(typed/"receipt.json")
        assert receipt==load(png/"receipt.json") and (typed/"answer.bin").read_bytes()==(png/"answer.bin").read_bytes()
        exact=receipt["stopped"] and (typed/"answer.bin").read_bytes()==b"".join(r.tile(c) for c in y)
        results.append(dict(row=index,expected=y,exact=exact,identical=True,**receipt))
    write_json(PUBLIC/"cli.json",results)
    print([(x["row"],x["exact"]) for x in results])


if __name__=="__main__":
    if "--register" in sys.argv:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,dict(examples=EXAMPLES,source_sha256=sha(Path(__file__)),cli_sha256=sha(Path(cli.__file__))))
    else:run()
