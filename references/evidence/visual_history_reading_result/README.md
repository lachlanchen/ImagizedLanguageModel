# Native visual history: complete auxiliary readouts

All 388 source responses below are an offline native text readout, not generated ILM answer images.
Question and reference-prefix pixels are inputs. All 216 visual/erased input calls are retained unchanged; duplicate PNGs share a hash path.
Raw strings, native decisions, source identities and all 648 cross-fitted first-piece readouts are in result.json.
This test supplies correct reference history. No autonomous visual-language model is qualified.

CMRC2018: Cui et al. (2019), CC BY-SA 4.0. Qwen3-VL: Alibaba, Apache-2.0.

| Group | Case | Prefix position | Route | Expected missing suffix | Raw source response | Exact after outer whitespace trim | Input |
|---|---:|---:|---|---|---|---|---|
| copy | 0 | 0 | visual | "春风吹过河岸" | "“春风吹过河岸”" | False | [PNG](inputs/bbe506b85338a280c485887fec5f37485debca771d097f8f4bac7e36914ca2b9.png) |
| copy | 0 | 0 | text | "春风吹过河岸" | "春风吹过河岸" | True | native text |
| copy | 0 | 3 | visual | "过河岸" | "过河岸" | True | [PNG](inputs/a05b7cdae00e78bdc5916e8ed5cd78f2d8a6d15b7cdfeea9f60c3768c9945634.png) |
| copy | 0 | 3 | text | "过河岸" | "过河岸" | True | native text |
| copy | 0 | 3 | erased | "过河岸" | "“春风吹过河岸”" | False | [PNG](inputs/bbe506b85338a280c485887fec5f37485debca771d097f8f4bac7e36914ca2b9.png) |
| copy | 0 | 5 | visual | "岸" | "岸" | True | [PNG](inputs/56b9581980d4bdd1d14c9353d17010653ef60b00e8efc8ae90d1f351896213eb.png) |
| copy | 0 | 5 | text | "岸" | "岸" | True | native text |
| copy | 0 | 6 | visual | "" | "“春风拂过河岸”" | False | [PNG](inputs/464a3ab32cba128b0b46ba04bdfa0aa391c6ae11ccd3af14dc30ed05acb7c628.png) |
| copy | 0 | 6 | text | "" | "" | True | native text |
| copy | 1 | 0 | visual | "秋雨落在山间" | "“秋雨落在山间”" | False | [PNG](inputs/6b67a4df25b4660dce5fef9219883aaaf543c1df4a30e5df1043aef3edbdbf48.png) |
| copy | 1 | 0 | text | "秋雨落在山间" | "秋雨落在山间" | True | native text |
| copy | 1 | 3 | visual | "在山间" | "在山间" | True | [PNG](inputs/a09b03058b7ba04ef0269278887033f98c56d29ea367dc7b23a2e87d49e6c13f.png) |
| copy | 1 | 3 | text | "在山间" | "在山间" | True | native text |
| copy | 1 | 3 | erased | "在山间" | "“秋雨落在山间”" | False | [PNG](inputs/6b67a4df25b4660dce5fef9219883aaaf543c1df4a30e5df1043aef3edbdbf48.png) |
| copy | 1 | 5 | visual | "间" | "间" | True | [PNG](inputs/3c3178c8dbf46d5a81a4b800bb9b64f0799c461c7dd72d500262cec937327977.png) |
| copy | 1 | 5 | text | "间" | "间" | True | native text |
| copy | 1 | 6 | visual | "" | "“秋雨落在山间”" | False | [PNG](inputs/d9ae3d37e3f209fc81a34814d097d1a9d924f39c861546e2d60be5dd1d726b3e.png) |
| copy | 1 | 6 | text | "" | "" | True | native text |
| copy | 2 | 0 | visual | "我们今天去图书馆。" | "我们今天去图书馆。" | True | [PNG](inputs/e3dcc235ae59914961eaae7ce01df4c91dfe4794beda53161a215b1efd6f13f8.png) |
| copy | 2 | 0 | text | "我们今天去图书馆。" | "去图书馆。" | False | native text |
| copy | 2 | 4 | visual | "去图书馆。" | "去图书馆。" | True | [PNG](inputs/8adc455e7b100488ad7f4b27d57bab17ac9ff241d463cbdcc4eabcedc31d79b7.png) |
| copy | 2 | 4 | text | "去图书馆。" | "去图书馆。" | True | native text |
| copy | 2 | 4 | erased | "去图书馆。" | "我们今天去图书馆。" | False | [PNG](inputs/e3dcc235ae59914961eaae7ce01df4c91dfe4794beda53161a215b1efd6f13f8.png) |
| copy | 2 | 8 | visual | "。" | "。" | True | [PNG](inputs/ca3beac3726fb5a4075136375d27bc5d1df9e545dd267d1dc44f3f7137c38a5b.png) |
| copy | 2 | 8 | text | "。" | "" | False | native text |
| copy | 2 | 9 | visual | "" | "我们今天去图书馆。" | False | [PNG](inputs/9dc2e3d41800809944a1b64a64e341883620e49666ad0570b05a005d258f68a3.png) |
| copy | 2 | 9 | text | "" | "" | True | native text |
| copy | 3 | 0 | visual | "他们明天去博物馆。" | "他们明天去博物馆。" | True | [PNG](inputs/a5ae443847f5251874aeb07ea5094106309c3b4442760ca87de44a56a9189d33.png) |
| copy | 3 | 0 | text | "他们明天去博物馆。" | "他们明天去博物馆。" | True | native text |
| copy | 3 | 4 | visual | "去博物馆。" | "去博物馆。" | True | [PNG](inputs/fe3688a898a36cf4d27f2336e9b4fcc0933fba8c9dc85349d5001699c9a3f76b.png) |
| copy | 3 | 4 | text | "去博物馆。" | "去博物馆。" | True | native text |
| copy | 3 | 4 | erased | "去博物馆。" | "他们明天去博物馆。" | False | [PNG](inputs/a5ae443847f5251874aeb07ea5094106309c3b4442760ca87de44a56a9189d33.png) |
| copy | 3 | 8 | visual | "。" | "。" | True | [PNG](inputs/97d3f74c3c458f1fc684a1386dea3ae94f18c38df327c1696f8d3716aaf25241.png) |
| copy | 3 | 8 | text | "。" | "" | False | native text |
| copy | 3 | 9 | visual | "" | "他们明天去博物馆。" | False | [PNG](inputs/a67242ca6b6f6aa934bfceb8ba4702a643282e762e0479ca12f199779292070b.png) |
| copy | 3 | 9 | text | "" | "" | True | native text |
| copy | 4 | 0 | visual | "The river is blue." | "\"The river is blue.\"" | False | [PNG](inputs/573c77609995a9498dcb7b4f7fb5180228a9d4fa31a0403b29df8958488648e5.png) |
| copy | 4 | 0 | text | "The river is blue." | "\"The river is blue.\"" | False | native text |
| copy | 4 | 9 | visual | "is blue." | "is blue." | True | [PNG](inputs/6ffa69894e3e971421ae25634b19616e06bdbc1734236a4b1c6b669147fa93d7.png) |
| copy | 4 | 9 | text | "is blue." | "is blue." | True | native text |
| copy | 4 | 9 | erased | "is blue." | "\"The river is blue.\"" | False | [PNG](inputs/573c77609995a9498dcb7b4f7fb5180228a9d4fa31a0403b29df8958488648e5.png) |
| copy | 4 | 17 | visual | "." | ".\"" | False | [PNG](inputs/6d355181470610b784048037e60cc7abf258b92d061454b25c7834bf0c019076.png) |
| copy | 4 | 17 | text | "." | "." | True | native text |
| copy | 4 | 18 | visual | "" | "." | False | [PNG](inputs/a83c18f08294f7d4051b9b52fe5615d0f1fc4c62094279ead60873db544392f5.png) |
| copy | 4 | 18 | text | "" | "" | True | native text |
| copy | 5 | 0 | visual | "The river is red." | "\"The river is red.\"" | False | [PNG](inputs/c5cb6428038942e9f130ac200b33dd38115feb735cf081a9555fad0cef8785c4.png) |
| copy | 5 | 0 | text | "The river is red." | "\"The river is red.\"" | False | native text |
| copy | 5 | 8 | visual | "r is red." | "r is red.\"" | False | [PNG](inputs/e06ee487477e3ef95312ded84dfabc97c191ec6b71cf378fcc578e93191bcafa.png) |
| copy | 5 | 8 | text | "r is red." | "r is red." | True | native text |
| copy | 5 | 8 | erased | "r is red." | "\"The river is red.\"" | False | [PNG](inputs/c5cb6428038942e9f130ac200b33dd38115feb735cf081a9555fad0cef8785c4.png) |
| copy | 5 | 16 | visual | "." | ".\"" | False | [PNG](inputs/9bbe458d7e194e019e0d4149aab821b2a32ce63ea95fae5ebad5e8c90aad06a7.png) |
| copy | 5 | 16 | text | "." | "." | True | native text |
| copy | 5 | 17 | visual | "" | "\"The river is red.\"" | False | [PNG](inputs/02e6ad27ea5e1ed4845972ee7f277ff90e2adc78aba47efe5acbe7d307c006be.png) |
| copy | 5 | 17 | text | "" | "" | True | native text |
| copy | 6 | 0 | visual | "A quiet garden." | "A quiet garden." | True | [PNG](inputs/71c9c9aa4acb0c40426d503f8c09646c6a4ea8af99760a60c801b4f5a246ce6d.png) |
| copy | 6 | 0 | text | "A quiet garden." | "A quiet garden." | True | native text |
| copy | 6 | 7 | visual | "garden." | "garden." | True | [PNG](inputs/1c05567f275c8a0b9fb46d00721c96210d81cea2d94e64126a1498ed9218fdcc.png) |
| copy | 6 | 7 | text | "garden." | "garden." | True | native text |
| copy | 6 | 7 | erased | "garden." | "A quiet garden." | False | [PNG](inputs/71c9c9aa4acb0c40426d503f8c09646c6a4ea8af99760a60c801b4f5a246ce6d.png) |
| copy | 6 | 14 | visual | "." | "A quiet garden" | False | [PNG](inputs/ae337be4a194cb26d65f1b0948b702c97902a26fd1f72a8b38735a0c9118b5d6.png) |
| copy | 6 | 14 | text | "." | "" | False | native text |
| copy | 6 | 15 | visual | "" | "A quiet garden." | False | [PNG](inputs/72fedb9ee933a6b3b0e853c9738fbc8fab1aac9dc6abfd279e799d55489ff4f0.png) |
| copy | 6 | 15 | text | "" | "" | True | native text |
| copy | 7 | 0 | visual | "A busy station." | "A busy station." | True | [PNG](inputs/b6d407b95b855aa382c01ff0679bf1366dd0397eb7b118e09d1f3ba66bc64fd4.png) |
| copy | 7 | 0 | text | "A busy station." | "A busy station." | True | native text |
| copy | 7 | 7 | visual | "station." | "station." | True | [PNG](inputs/4122f51b2516ca051a0c8e99c78f0ef03a41b8cd6bb4ea6720550b4870df43c9.png) |
| copy | 7 | 7 | text | "station." | "station." | True | native text |
| copy | 7 | 7 | erased | "station." | "A busy station." | False | [PNG](inputs/b6d407b95b855aa382c01ff0679bf1366dd0397eb7b118e09d1f3ba66bc64fd4.png) |
| copy | 7 | 14 | visual | "." | "A busy station" | False | [PNG](inputs/2eb36dd269d51981bbe4e1e6ca01fef1979250824dd0280a3261643d6d9db3de.png) |
| copy | 7 | 14 | text | "." | "" | False | native text |
| copy | 7 | 15 | visual | "" | "A busy station." | False | [PNG](inputs/68217e12cd768965b73282a33c7477b73a4836964f46a8b801c26e6d16d69f2a.png) |
| copy | 7 | 15 | text | "" | "" | True | native text |
| copy | 8 | 0 | visual | "702945" | "702945" | True | [PNG](inputs/016ccdd9fdc897d9ea2c1ba1808bb554f165b53de50e7640f7b96f5975dc213d.png) |
| copy | 8 | 0 | text | "702945" | "702945" | True | native text |
| copy | 8 | 3 | visual | "945" | "945" | True | [PNG](inputs/581136c5a85c7b5a952f5067b182c89b06bb6ed509dfbaed9529a50da33a0454.png) |
| copy | 8 | 3 | text | "945" | "945" | True | native text |
| copy | 8 | 3 | erased | "945" | "702945" | False | [PNG](inputs/016ccdd9fdc897d9ea2c1ba1808bb554f165b53de50e7640f7b96f5975dc213d.png) |
| copy | 8 | 5 | visual | "5" | "5" | True | [PNG](inputs/af68ef080ef8812cce2d0fcb3ad366860aa3cb0d47fd73d8264205776b6c3259.png) |
| copy | 8 | 5 | text | "5" | "5" | True | native text |
| copy | 8 | 6 | visual | "" | "702945" | False | [PNG](inputs/9b9b2f9e39349ca2713ee9b222e30d7b59cd86c6a15708c5081652fcfdfa36d4.png) |
| copy | 8 | 6 | text | "" | "" | True | native text |
| copy | 9 | 0 | visual | "702946" | "702946" | True | [PNG](inputs/881ffc14052a1d4a46b4ece7cbb848cdb328df7feee1804c1157531bccd6b999.png) |
| copy | 9 | 0 | text | "702946" | "702946" | True | native text |
| copy | 9 | 3 | visual | "946" | "946" | True | [PNG](inputs/049b3391a33d30d944f137978bcbbbbb49a71d834e91161fdfc6d3569a34fbf1.png) |
| copy | 9 | 3 | text | "946" | "946" | True | native text |
| copy | 9 | 3 | erased | "946" | "702946" | False | [PNG](inputs/881ffc14052a1d4a46b4ece7cbb848cdb328df7feee1804c1157531bccd6b999.png) |
| copy | 9 | 5 | visual | "6" | "6" | True | [PNG](inputs/e9b7c96e090cc4c12d33b1f8d958323ddb1e31d1825478d2321b928c8571d724.png) |
| copy | 9 | 5 | text | "6" | "6" | True | native text |
| copy | 9 | 6 | visual | "" | "702946" | False | [PNG](inputs/2e3d8cc988b1df1ae1742b6770be7a79dae52f4111010eb41c92f23ee766d252.png) |
| copy | 9 | 6 | text | "" | "" | True | native text |
| copy | 10 | 0 | visual | "381607" | "381607" | True | [PNG](inputs/4851ff4683073c887a4990ee5a006a57feed431a860cd645022b866a0f08042e.png) |
| copy | 10 | 0 | text | "381607" | "381607" | True | native text |
| copy | 10 | 3 | visual | "607" | "607" | True | [PNG](inputs/377b1fa9120947f7254ab3d0e44e1e275b15381d1749c049b3181a3c2285f0ef.png) |
| copy | 10 | 3 | text | "607" | "607" | True | native text |
| copy | 10 | 3 | erased | "607" | "381607" | False | [PNG](inputs/4851ff4683073c887a4990ee5a006a57feed431a860cd645022b866a0f08042e.png) |
| copy | 10 | 5 | visual | "7" | "7" | True | [PNG](inputs/3a4c74af64433bc79a09e3814448600b54a6041947bda45efe27da5db5db5ade.png) |
| copy | 10 | 5 | text | "7" | "7" | True | native text |
| copy | 10 | 6 | visual | "" | "381607" | False | [PNG](inputs/b7e9b96e125e09f7e535a02e4a7a9343f270bcef93181825f271618d7a2e1626.png) |
| copy | 10 | 6 | text | "" | "" | True | native text |
| copy | 11 | 0 | visual | "381608" | "381608" | True | [PNG](inputs/246d0afea1587705f41c3ce60497419dc2b1536fdad30e49e202a5056569fb26.png) |
| copy | 11 | 0 | text | "381608" | "381608" | True | native text |
| copy | 11 | 3 | visual | "608" | "608" | True | [PNG](inputs/fd0b71f7415b962348b36a34f0a8ae28f72ce872e36ca36410050f320eb2ed06.png) |
| copy | 11 | 3 | text | "608" | "608" | True | native text |
| copy | 11 | 3 | erased | "608" | "381608" | False | [PNG](inputs/246d0afea1587705f41c3ce60497419dc2b1536fdad30e49e202a5056569fb26.png) |
| copy | 11 | 5 | visual | "8" | "8" | True | [PNG](inputs/f5867bcfaa3ec257810a92cb72d8320f5f61e9e4a416f1ee1a8ccd9b7370857a.png) |
| copy | 11 | 5 | text | "8" | "8" | True | native text |
| copy | 11 | 6 | visual | "" | "381608" | False | [PNG](inputs/17adc82e67c6a525754a4b22bfae1da3eb22dd763580579918e8ef6045058c52.png) |
| copy | 11 | 6 | text | "" | "" | True | native text |
| training | 0 | 0 | visual | "冯兴" | "冯兴" | True | [PNG](inputs/4ee32f61b9f0511165d0d725c385026ed62bfd9c65bd15c29f233c83e7159dbe.png) |
| training | 0 | 0 | text | "冯兴" | "冯兴" | True | native text |
| training | 0 | 1 | visual | "兴" | "冯兴" | False | [PNG](inputs/44bcd9d610c8be349ebd1473a630b31524c0da42493858217890274b47969a5f.png) |
| training | 0 | 1 | text | "兴" | "兴" | True | native text |
| training | 0 | 1 | erased | "兴" | "冯兴" | False | [PNG](inputs/4ee32f61b9f0511165d0d725c385026ed62bfd9c65bd15c29f233c83e7159dbe.png) |
| training | 0 | 2 | visual | "" | "冯兴" | False | [PNG](inputs/d69edfbfd2c6592a3c7c68c3497af1aab0af7e02f810b85343f52400f659afc3.png) |
| training | 0 | 2 | text | "" | "是8世纪越南反唐起义军的军事首领" | False | native text |
| training | 1 | 0 | visual | "杜英翰" | "杜英翰" | True | [PNG](inputs/11826f582abbfab0e98cfe9fe00b6848367623c8a75f28d456ee1a941ef4aea7.png) |
| training | 1 | 0 | text | "杜英翰" | "杜英翰" | True | native text |
| training | 1 | 1 | visual | "英翰" | "杜英翰" | False | [PNG](inputs/3f8fe7814449c9107c9609d7cf7783f973d24cb3e785f44258141ca62e7330fc.png) |
| training | 1 | 1 | text | "英翰" | "英翰" | True | native text |
| training | 1 | 1 | erased | "英翰" | "杜英翰" | False | [PNG](inputs/11826f582abbfab0e98cfe9fe00b6848367623c8a75f28d456ee1a941ef4aea7.png) |
| training | 1 | 2 | visual | "翰" | "翰" | True | [PNG](inputs/48aae5093a5e4c0c5c66554b859e7234e7d7aa9562adc9af16c8b32f41352f8f.png) |
| training | 1 | 2 | text | "翰" | "翰" | True | native text |
| training | 1 | 3 | visual | "" | "杜英翰" | False | [PNG](inputs/79c88cd4fd0699b856a87c3851131e2da7c8dd8749ed2d1cf2100b98dfa75c39.png) |
| training | 1 | 3 | text | "" | "率众起义" | False | native text |
| training | 2 | 0 | visual | "公元前201年" | "公元前201年" | True | [PNG](inputs/1bc255d88a20eb3cc249154e0adbbbccdaa322a0c734ef7242bdea4c8bccad1e.png) |
| training | 2 | 0 | text | "公元前201年" | "公元前201年" | True | native text |
| training | 2 | 3 | visual | "201年" | "201年" | True | [PNG](inputs/73994d093f5753d21180eef6ce5219101ba3d87f098b9464103278ce208bb830.png) |
| training | 2 | 3 | text | "201年" | "201年" | True | native text |
| training | 2 | 3 | erased | "201年" | "公元前201年" | False | [PNG](inputs/1bc255d88a20eb3cc249154e0adbbbccdaa322a0c734ef7242bdea4c8bccad1e.png) |
| training | 2 | 6 | visual | "年" | "公元前201" | False | [PNG](inputs/1b558b40f1b693084d18fb78ae972d8d15b9ded0773a65912336493bacbd7ba0.png) |
| training | 2 | 6 | text | "年" | "年" | True | native text |
| training | 2 | 7 | visual | "" | "公元前201年" | False | [PNG](inputs/c8ba745bde40fcdf321a4583f893bace4cfc66c4d99e505ec5434f321f80d123.png) |
| training | 2 | 7 | text | "" | "可能发生在公元前201年" | False | native text |
| training | 3 | 0 | visual | "爱琴海沿岸的希腊城邦" | "小亚细亚和帕加马问题" | False | [PNG](inputs/13673973563eb938f6e69c8157f6399bb32874df110ab9fa92081fb2d66a97bf.png) |
| training | 3 | 0 | text | "爱琴海沿岸的希腊城邦" | "爱琴海沿岸的希腊城邦" | True | native text |
| training | 3 | 5 | visual | "的希腊城邦" | "爱琴海沿岸" | False | [PNG](inputs/735d62d885de54e580fbcb1d3c594e730bfd693708e4ed08c4cc09f97b7fd511.png) |
| training | 3 | 5 | text | "的希腊城邦" | "的希腊城邦" | True | native text |
| training | 3 | 5 | erased | "的希腊城邦" | "小亚细亚和帕加马问题" | False | [PNG](inputs/13673973563eb938f6e69c8157f6399bb32874df110ab9fa92081fb2d66a97bf.png) |
| training | 3 | 9 | visual | "邦" | "爱琴海沿岸的希腊城" | False | [PNG](inputs/0a987472d2de661f6e2511f6b1cfb099df26efcc9d0901142b61332b6193d4fd.png) |
| training | 3 | 9 | text | "邦" | "邦" | True | native text |
| training | 3 | 10 | visual | "" | "爱琴海沿岸的希腊城邦" | False | [PNG](inputs/eb982b7258b379e5f721b46054be6c98f1c535c9dd3890a3ada7b829683a5d96.png) |
| training | 3 | 10 | text | "" | "因而爆发了克里特战争" | False | native text |
| training | 4 | 0 | visual | "已灭绝的银杏类植物" | "乌马托鳞片属是一类已灭绝的银杏类植物，属于乌马托鳞片科（Umaltolepidiaceae），生存于中侏罗世到早白垩世。其雌性繁殖器官包括一个苞片和短柄，短柄基部具有" | False | [PNG](inputs/173df570750be2198a50c8e02be2709caa8e58505879bb936ccf4c9950b3ecb3.png) |
| training | 4 | 0 | text | "已灭绝的银杏类植物" | "已灭绝的银杏类植物，属于乌马托鳞片科，生存于中侏罗世到早白垩世，具有特定的雌性繁殖器官和线性、长舌形的营养叶，且在多个亚洲地区地层中均有发现。" | False | native text |
| training | 4 | 4 | visual | "银杏类植物" | "已灭绝的银杏类植物" | False | [PNG](inputs/45674b7d0127f082e92225fc88391a11be7bb87f3951e4b70f9c3787ee38319a.png) |
| training | 4 | 4 | text | "银杏类植物" | "银杏类植物，属于乌马托鳞片科，生存于中侏罗世到早白垩世，雌性繁殖器官包括苞片和短柄，短柄基部具鳞片叶，苞片顶生可能含两个裂片，胚珠生于腋部；营养叶叶" | False | native text |
| training | 4 | 4 | erased | "银杏类植物" | "乌马托鳞片属是一类已灭绝的银杏类植物，属于乌马托鳞片科（Umaltolepidiaceae），生存于中侏罗世到早白垩世。其雌性繁殖器官包括一个苞片和短柄，短柄基部具有" | False | [PNG](inputs/173df570750be2198a50c8e02be2709caa8e58505879bb936ccf4c9950b3ecb3.png) |
| training | 4 | 8 | visual | "物" | "已灭绝的银杏类植" | False | [PNG](inputs/0cf487499fc0163bea0a32c3d8134d461db2e98b81c8fb8c5868fd768994c3f8.png) |
| training | 4 | 8 | text | "物" | "物，属于乌马托鳞片科，生存于中侏罗世到早白垩世，雌性繁殖器官具苞片和短柄，短柄基部有鳞片叶，苞片顶生可能含两裂片，胚珠生于腋部；营养叶线形或长" | False | native text |
| training | 4 | 9 | visual | "" | "已灭绝的银杏类植物" | False | [PNG](inputs/9479a542f177714319d7d138120eed49dbb358349ae8b47d242fa26e3f16cbbf.png) |
| training | 4 | 9 | text | "" | "属于乌马托鳞片科，生存于中侏罗世到早白垩世，雌性繁殖器官具苞片和短柄，短柄基部有鳞片叶，苞片顶生可能含两裂片，胚珠生于腋部；营养叶线形或长舌形" | False | native text |
| training | 5 | 0 | visual | "中侏罗世到早白垩世" | "中侏罗世到早白垩世" | True | [PNG](inputs/9461a624b9dbd663e9a860ec859677a5a3458a4c681cec44cc266a527e4d32c0.png) |
| training | 5 | 0 | text | "中侏罗世到早白垩世" | "中侏罗世到早白垩世" | True | native text |
| training | 5 | 4 | visual | "到早白垩世" | "到早白垩世" | True | [PNG](inputs/d096e067b7f5e23e676c6c859d0b37ae12416d05f5685d4fe8518a17e26b3bd8.png) |
| training | 5 | 4 | text | "到早白垩世" | "到早白垩世" | True | native text |
| training | 5 | 4 | erased | "到早白垩世" | "中侏罗世到早白垩世" | False | [PNG](inputs/9461a624b9dbd663e9a860ec859677a5a3458a4c681cec44cc266a527e4d32c0.png) |
| training | 5 | 8 | visual | "世" | "世" | True | [PNG](inputs/86f7247d3dabd89578a787faac600993cd30ee894cf58b8c77ee00cb9aff2faf.png) |
| training | 5 | 8 | text | "世" | "世" | True | native text |
| training | 5 | 9 | visual | "" | "中侏罗世到早白垩世" | False | [PNG](inputs/48350add5e4f0a46f87ccc2328887cdf67d7cd549f1dd505ddc5a07efffb696f.png) |
| training | 5 | 9 | text | "" | "在西伯利亚的晚侏罗世和早白垩世地层，中国东北和华北的早白垩世地层，以及中国河南的中侏罗世地层中均有发现" | False | native text |
| training | 6 | 0 | visual | "美国纽约市" | "美国纽约市" | True | [PNG](inputs/87870b03ae7310eba047d42932eef1c3e88ec543ff18bbe5f04b0427c1a427f9.png) |
| training | 6 | 0 | text | "美国纽约市" | "美国纽约市" | True | native text |
| training | 6 | 2 | visual | "纽约市" | "美国纽约市" | False | [PNG](inputs/205b5c35577abb4d0039dba2769c8d41ee72ae1746ee31941fb574c235ad7064.png) |
| training | 6 | 2 | text | "纽约市" | "纽约市" | True | native text |
| training | 6 | 2 | erased | "纽约市" | "美国纽约市" | False | [PNG](inputs/87870b03ae7310eba047d42932eef1c3e88ec543ff18bbe5f04b0427c1a427f9.png) |
| training | 6 | 4 | visual | "市" | "美国纽约" | False | [PNG](inputs/b2ab984770ac14dd2a764c6c4bc544309e7898cca316762a9b682b2da67d93bf.png) |
| training | 6 | 4 | text | "市" | "市" | True | native text |
| training | 6 | 5 | visual | "" | "美国纽约市" | False | [PNG](inputs/4a7c4affcd8dcc6b20a6daff1853e545bc78f8788d9252403febcd0930f85c0b.png) |
| training | 6 | 5 | text | "" | "于美国纽约市出生" | False | native text |
| training | 7 | 0 | visual | "9岁" | "9岁时" | False | [PNG](inputs/c59a8e74834b534eecd00be419c3fb2d8b18b5f5dc628b0bea0644bdf939ffb1.png) |
| training | 7 | 0 | text | "9岁" | "9岁时" | False | native text |
| training | 7 | 1 | visual | "岁" | "岁" | True | [PNG](inputs/bb85ea9f948eb3adb71eccb8abe686fb2b64b16b955cc586d7148ac71fa0634d.png) |
| training | 7 | 1 | text | "岁" | "岁" | True | native text |
| training | 7 | 1 | erased | "岁" | "9岁时" | False | [PNG](inputs/c59a8e74834b534eecd00be419c3fb2d8b18b5f5dc628b0bea0644bdf939ffb1.png) |
| training | 7 | 2 | visual | "" | "9岁" | False | [PNG](inputs/f15ea1e6728b4e14d32034b2a210d8f334fdf2190664be8860d5830ce7c3bd0d.png) |
| training | 7 | 2 | text | "" | "时" | False | native text |
| training | 8 | 0 | visual | "全程：$20.8" | "$20.8" | False | [PNG](inputs/9fea1d19f6ab0c2c0b077954cf4f63afcf6fad5c5353f3f3e1ef27e43b262aef.png) |
| training | 8 | 0 | text | "全程：$20.8" | "$20.8" | False | native text |
| training | 8 | 4 | visual | "20.8" | "20.8" | True | [PNG](inputs/deea0501059088d2b0b4e647bc8929af1e46a371fcda87e71111d9e8e788bfd5.png) |
| training | 8 | 4 | text | "20.8" | "20.8" | True | native text |
| training | 8 | 4 | erased | "20.8" | "$20.8" | False | [PNG](inputs/9fea1d19f6ab0c2c0b077954cf4f63afcf6fad5c5353f3f3e1ef27e43b262aef.png) |
| training | 8 | 7 | visual | "8" | "$20." | False | [PNG](inputs/0a642816d831df788e20b3260a8f46cf7681fda329ef9951538013fd35811e82.png) |
| training | 8 | 7 | text | "8" | "8" | True | native text |
| training | 8 | 8 | visual | "" | "$20.8" | False | [PNG](inputs/eea64ffcfeb665bb7e43f85b9a64aad37d159926b2dac95c5c5f8542d2cf1af9.png) |
| training | 8 | 8 | text | "" | "乘客登上本线后150分钟内以同一张八达通卡转乘以下路线，或从以下路线登车后150分钟内以同一张八达通卡转乘本线，次程可获车资折扣优惠：本线主要是由66M、6" | False | native text |
| training | 9 | 0 | visual | "由九巴独自经营" | "九巴" | False | [PNG](inputs/be30ce2abc9b49ecbe69f691530a21307cf229a3c536bae1d4b14e41a6e53e2c.png) |
| training | 9 | 0 | text | "由九巴独自经营" | "九巴独自经营" | False | native text |
| training | 9 | 3 | visual | "独自经营" | "独自经营" | True | [PNG](inputs/e7bc5af5a84c19199d404c83e42a44a1126a3fce518d9a75adac25b0d9cfa95f.png) |
| training | 9 | 3 | text | "独自经营" | "独自经营" | True | native text |
| training | 9 | 3 | erased | "独自经营" | "九巴" | False | [PNG](inputs/be30ce2abc9b49ecbe69f691530a21307cf229a3c536bae1d4b14e41a6e53e2c.png) |
| training | 9 | 6 | visual | "营" | "营的" | False | [PNG](inputs/c34c13cb00c48a1c079d817ed0066053fe3cbbf165a1fda82714d1381c4cebb1.png) |
| training | 9 | 6 | text | "营" | "营" | True | native text |
| training | 9 | 7 | visual | "" | "由九巴独自经营" | False | [PNG](inputs/6ac5db1ed07d421db657a268bd77526352f72ee790327457869f5d0b9bff646c.png) |
| training | 9 | 7 | text | "" | "由九巴独自经营" | False | native text |
| training | 10 | 0 | visual | "7.13平方公里" | "7.13平方公里" | True | [PNG](inputs/8cbdd654028fc82205fabefac1704d235056cb67c7aa77c7844902f3afe0cdc9.png) |
| training | 10 | 0 | text | "7.13平方公里" | "7.13平方公里" | True | native text |
| training | 10 | 4 | visual | "平方公里" | "7.13平方公里" | False | [PNG](inputs/a10388b12400ad914e71ac37fface1c0737a8e06ef0a18f81e1a37655ca50f16.png) |
| training | 10 | 4 | text | "平方公里" | "平方公里" | True | native text |
| training | 10 | 4 | erased | "平方公里" | "7.13平方公里" | False | [PNG](inputs/8cbdd654028fc82205fabefac1704d235056cb67c7aa77c7844902f3afe0cdc9.png) |
| training | 10 | 7 | visual | "里" | "里" | True | [PNG](inputs/f85462e1d5d9481ad5000ea13fd7f8411c0b3266534b9106a34cb0eeff3f2d53.png) |
| training | 10 | 7 | text | "里" | "里" | True | native text |
| training | 10 | 8 | visual | "" | "7.13平方公里" | False | [PNG](inputs/1513ffd3e608603f16cafaa271cec81ef34388749eac9a18100b71b69156ffbc.png) |
| training | 10 | 8 | text | "" | "" | True | native text |
| training | 11 | 0 | visual | "0.85万" | "0.85万" | True | [PNG](inputs/689589e2c1e1bde3daa02cf2d4ae6d02a8c2f831730e5a41a23293a6dca32291.png) |
| training | 11 | 0 | text | "0.85万" | "0.85万" | True | native text |
| training | 11 | 2 | visual | "85万" | "0.85万" | False | [PNG](inputs/a5ec7d1e5e95561a3391f069dba484f8cdb401c5e1def03d485c458c395b27c6.png) |
| training | 11 | 2 | text | "85万" | "85万" | True | native text |
| training | 11 | 2 | erased | "85万" | "0.85万" | False | [PNG](inputs/689589e2c1e1bde3daa02cf2d4ae6d02a8c2f831730e5a41a23293a6dca32291.png) |
| training | 11 | 4 | visual | "万" | "万" | True | [PNG](inputs/48af467c278ef86b84c97e837dfc140e66546371f4491a463d6054188ba5543d.png) |
| training | 11 | 4 | text | "万" | "万" | True | native text |
| training | 11 | 5 | visual | "" | "0.85万" | False | [PNG](inputs/515fb2b48da98ce50a6d899b62d8f5c4baeb58475133ee47ab7233efe3991f31.png) |
| training | 11 | 5 | text | "" | "" | True | native text |
| training | 12 | 0 | visual | "美国女演员，曾当过模特儿" | "美国女演员" | False | [PNG](inputs/ad606d6199e0581a20e1d752c6654691435b9a98ace000c40bddb20d25b2b493.png) |
| training | 12 | 0 | text | "美国女演员，曾当过模特儿" | "美国女演员，曾当过模特儿" | True | native text |
| training | 12 | 6 | visual | "曾当过模特儿" | "曾当过模特儿" | True | [PNG](inputs/f01b063ffa8affa0b7e72be397a25c309937fc21d24096f172fe04925be4a65e.png) |
| training | 12 | 6 | text | "曾当过模特儿" | "曾当过模特儿" | True | native text |
| training | 12 | 6 | erased | "曾当过模特儿" | "美国女演员" | False | [PNG](inputs/ad606d6199e0581a20e1d752c6654691435b9a98ace000c40bddb20d25b2b493.png) |
| training | 12 | 11 | visual | "儿" | "美国女演员，曾当过模特" | False | [PNG](inputs/b654970d700a525c5b50a5b1d139ad639636771672a3eb32971644c3e34cb5e2.png) |
| training | 12 | 11 | text | "儿" | "。" | False | native text |
| training | 12 | 12 | visual | "" | "美国女演员，曾当过模特儿" | False | [PNG](inputs/e181444edc927f8652e5a8fbe5cff5aef3cace7f77924e1e11c6e2558a530000.png) |
| training | 12 | 12 | text | "" | "她在荷里活电影《德州电锯杀人狂》、《TheIllusionist》和电视连续剧《7thHeaven》的演出最为人熟悉。" | False | native text |
| training | 13 | 0 | visual | "在美国明尼苏达州伊利出生" | "美国明尼苏达州伊利" | False | [PNG](inputs/e8caadc88aa3bd3a97a2a99b6ec4cb6e94e83d20e03d0ee7bfd62359214a259e.png) |
| training | 13 | 0 | text | "在美国明尼苏达州伊利出生" | "美国明尼苏达州伊利" | False | native text |
| training | 13 | 6 | visual | "达州伊利出生" | "达州伊利" | False | [PNG](inputs/68fd617e8fa0e8200d9b5d6a0ef7c4ead0a410e703089aa09ab65b822580b89b.png) |
| training | 13 | 6 | text | "达州伊利出生" | "达州伊利" | False | native text |
| training | 13 | 6 | erased | "达州伊利出生" | "美国明尼苏达州伊利" | False | [PNG](inputs/e8caadc88aa3bd3a97a2a99b6ec4cb6e94e83d20e03d0ee7bfd62359214a259e.png) |
| training | 13 | 11 | visual | "生" | "伊利出" | False | [PNG](inputs/039531d8121f0efbdff799127d4b52a1f193c808be355b61b71764e9b772d523.png) |
| training | 13 | 11 | text | "生" | "生" | True | native text |
| training | 13 | 12 | visual | "" | "在美国明尼苏达州伊利出生" | False | [PNG](inputs/514c39ea8296bdae9a474a49ac8737505110286b88d5eb087643cc703ecdac3c.png) |
| training | 13 | 12 | text | "" | "在美国明尼苏达州伊利" | False | native text |
| training | 14 | 0 | visual | "加利福尼亚州洛杉矶" | "加利福尼亚州洛杉矶" | True | [PNG](inputs/51740e6a9d6aa5fc37ffee9b9a1f24db89729496a79ef27f5ec87b6b9085ed92.png) |
| training | 14 | 0 | text | "加利福尼亚州洛杉矶" | "加利福尼亚州洛杉矶" | True | native text |
| training | 14 | 4 | visual | "亚州洛杉矶" | "亚州洛杉矶" | True | [PNG](inputs/037b58671d5ed9d058b9aa921158e0c7f1ffd756e639f1abfdac644ec3359511.png) |
| training | 14 | 4 | text | "亚州洛杉矶" | "亚州洛杉矶" | True | native text |
| training | 14 | 4 | erased | "亚州洛杉矶" | "加利福尼亚州洛杉矶" | False | [PNG](inputs/51740e6a9d6aa5fc37ffee9b9a1f24db89729496a79ef27f5ec87b6b9085ed92.png) |
| training | 14 | 8 | visual | "矶" | "洛杉矶" | False | [PNG](inputs/de828ac9105974ee4b0613ecd319380e30cb96d8aa95d083163611da3263ea0e.png) |
| training | 14 | 8 | text | "矶" | "矶" | True | native text |
| training | 14 | 9 | visual | "" | "加利福尼亚州洛杉矶" | False | [PNG](inputs/b55feb03da29c66afeb502663cd41661790e2306a1d124d149aa785788dd5042.png) |
| training | 14 | 9 | text | "" | "" | True | native text |
| training | 15 | 0 | visual | "2011年球队选择权" | "加上2011年球队选择权的合约。" | False | [PNG](inputs/d6f9ab37306c815b5a34fddb37718c3bbb208d8b46e31afe38dd407120f92f91.png) |
| training | 15 | 0 | text | "2011年球队选择权" | "2011年球队选择权" | True | native text |
| training | 15 | 5 | visual | "球队选择权" | "球队选择权" | True | [PNG](inputs/84f016a3b25f779951a70fa46ea4331a757e98931a40c3caa6fed1e890a542bf.png) |
| training | 15 | 5 | text | "球队选择权" | "球队选择权" | True | native text |
| training | 15 | 5 | erased | "球队选择权" | "加上2011年球队选择权的合约。" | False | [PNG](inputs/d6f9ab37306c815b5a34fddb37718c3bbb208d8b46e31afe38dd407120f92f91.png) |
| training | 15 | 9 | visual | "权" | "2011年球队选择" | False | [PNG](inputs/72587051c8d1bac0f3247b574f7b953fad32fb0610e2860c42ae73c5e12d4a2e.png) |
| training | 15 | 9 | text | "权" | "权" | True | native text |
| training | 15 | 10 | visual | "" | "2011年球队选择权" | False | [PNG](inputs/eaa9018c10abc047688c9db8c143c6b80f5aef5d4ea97940e57e2bbb4ba32278.png) |
| training | 15 | 10 | text | "" | "2011年球队选择权" | False | native text |
| training | 16 | 0 | visual | "1984年8月21日出生" | "1984年" | False | [PNG](inputs/7f9e200595a246e4040ccd02568f57e2aa8578c7a2b3d7984a90c96196d6c46c.png) |
| training | 16 | 0 | text | "1984年8月21日出生" | "1984年" | False | native text |
| training | 16 | 6 | visual | "月21日出生" | "21日" | False | [PNG](inputs/3d211c65c90c1fb017d6905ad7f57a58c3cf94854473c843618fe89f98b6cb01.png) |
| training | 16 | 6 | text | "月21日出生" | "月21日" | False | native text |
| training | 16 | 6 | erased | "月21日出生" | "1984年" | False | [PNG](inputs/7f9e200595a246e4040ccd02568f57e2aa8578c7a2b3d7984a90c96196d6c46c.png) |
| training | 16 | 11 | visual | "生" | "出生于1984年8月21日" | False | [PNG](inputs/4e8bb54e367370c4ec75b3569d2d15c2ea18fe835b1886dafd7c4ec06d51bbd5.png) |
| training | 16 | 11 | text | "生" | "生于1984年" | False | native text |
| training | 16 | 12 | visual | "" | "1984年8月21日出生" | False | [PNG](inputs/f8e3c7d0747dcd69eb5886b6087ed51bf7dbf7da199fa126379616c390cf3df2.png) |
| training | 16 | 12 | text | "" | "1984年" | False | native text |
| training | 17 | 0 | visual | "美国维吉尼亚州的诺福克" | "美国维吉尼亚州的诺福克" | True | [PNG](inputs/1cd6fb3a230d6d084edeb1830176dbc1b8664e67e96612e6ee13ad9697e47c6d.png) |
| training | 17 | 0 | text | "美国维吉尼亚州的诺福克" | "美国维吉尼亚州的诺福克" | True | native text |
| training | 17 | 5 | visual | "亚州的诺福克" | "亚州的诺福克" | True | [PNG](inputs/696a9c915d301d9d5c1005f523b68f8fedf75012a3080c93c3e94a30ba79302d.png) |
| training | 17 | 5 | text | "亚州的诺福克" | "亚州的诺福克" | True | native text |
| training | 17 | 5 | erased | "亚州的诺福克" | "美国维吉尼亚州的诺福克" | False | [PNG](inputs/1cd6fb3a230d6d084edeb1830176dbc1b8664e67e96612e6ee13ad9697e47c6d.png) |
| training | 17 | 10 | visual | "克" | "克" | True | [PNG](inputs/1a7763ee82731ce3133b8c0c86ca7f8279d6af8a1a9fe3bcc8c333f9892b5a79.png) |
| training | 17 | 10 | text | "克" | "克" | True | native text |
| training | 17 | 11 | visual | "" | "美国维吉尼亚州的诺福克" | False | [PNG](inputs/8a500b3393b7feb848454d2eea6689ff62c7a69fee3800df3b631e41b6be9b1e.png) |
| training | 17 | 11 | text | "" | "美国维吉尼亚州的诺福克" | False | native text |
| training | 18 | 0 | visual | "两种" | "两种" | True | [PNG](inputs/6a3504333ee071d1e9371c69aa5ad42cca31bd291e5093cd98a938ea8b9037a6.png) |
| training | 18 | 0 | text | "两种" | "两种" | True | native text |
| training | 18 | 1 | visual | "种" | "种" | True | [PNG](inputs/b191115e3ec8c10c061134d3c0a2784e8c143fd42aba9fa9441cd5919dcd67a5.png) |
| training | 18 | 1 | text | "种" | "种" | True | native text |
| training | 18 | 1 | erased | "种" | "两种" | False | [PNG](inputs/6a3504333ee071d1e9371c69aa5ad42cca31bd291e5093cd98a938ea8b9037a6.png) |
| training | 18 | 2 | visual | "" | "两种" | False | [PNG](inputs/92e478cd98ff6f18ba0df0d2861efc6b41f781827d97e24d0b40a1eda9b47d38.png) |
| training | 18 | 2 | text | "" | "分别是高山白腹鼠以及刺鼠" | False | native text |
| training | 19 | 0 | visual | "由其台湾俗名\"国姓鼠\"而来" | "是由其台湾俗名\"国姓鼠\"而来。" | False | [PNG](inputs/36e5f6a3c5e17119c4859b42c9c79e67b0a274af1fe492f7c1fe4948407e67ad.png) |
| training | 19 | 0 | text | "由其台湾俗名\"国姓鼠\"而来" | "由其台湾俗名\"国姓鼠\"而来" | True | native text |
| training | 19 | 6 | visual | "\"国姓鼠\"而来" | "\"国姓鼠\"而来" | True | [PNG](inputs/7fb2b7174e41a02a0a8098aa390c6322e89111e6e5327e1ba25bc030843f8e17.png) |
| training | 19 | 6 | text | "\"国姓鼠\"而来" | "“国姓鼠”而来" | False | native text |
| training | 19 | 6 | erased | "\"国姓鼠\"而来" | "是由其台湾俗名\"国姓鼠\"而来。" | False | [PNG](inputs/36e5f6a3c5e17119c4859b42c9c79e67b0a274af1fe492f7c1fe4948407e67ad.png) |
| training | 19 | 12 | visual | "来" | "由其台湾俗名\"国姓鼠\"而来" | False | [PNG](inputs/b2e310fdb70a9323bc9356d2aa0e8415bb3a71910051675441e0f9f312720237.png) |
| training | 19 | 12 | text | "来" | "来" | True | native text |
| training | 19 | 13 | visual | "" | "由其台湾俗名\"国姓鼠\"而来" | False | [PNG](inputs/c0fa1d1d5bd61dc4d54399cfec675d4a05b4758ade16974bb813c4a018a7e016.png) |
| training | 19 | 13 | text | "" | "" | True | native text |
| training | 20 | 0 | visual | "科举制度" | "科举制度" | True | [PNG](inputs/29355ef78a4f454736cb59a33e563588c3dcdd089408fffeb63870ba51de1ebc.png) |
| training | 20 | 0 | text | "科举制度" | "科举制度" | True | native text |
| training | 20 | 2 | visual | "制度" | "制度" | True | [PNG](inputs/7edea9057ffaf2c70de4e9db9ee073ad76fc56fb4a5280a5cbdbcc643cf54a90.png) |
| training | 20 | 2 | text | "制度" | "制度" | True | native text |
| training | 20 | 2 | erased | "制度" | "科举制度" | False | [PNG](inputs/29355ef78a4f454736cb59a33e563588c3dcdd089408fffeb63870ba51de1ebc.png) |
| training | 20 | 3 | visual | "度" | "的特殊宴会" | False | [PNG](inputs/c0f90de0275463cfa1654e0e1d0d4efe0031180f70413e35545e09943b88e6c5.png) |
| training | 20 | 3 | text | "度" | "下的特殊宴会" | False | native text |
| training | 20 | 4 | visual | "" | "科举制度" | False | [PNG](inputs/fe394a1aa224da4a901c8e2e3f7014f5d383fa3f7012565b914aed6b9ec59862.png) |
| training | 20 | 4 | text | "" | "始于唐代，明清沿袭，清代由巡抚主持，宴请新科举人及考官，唱《鹿鸣》诗并跳魁星舞，兼具饯行、励志与礼仪功能，南宋后繁文缛节加重地方负担，几近废礼。" | False | native text |
| training | 21 | 0 | visual | "唐代" | "唐代" | True | [PNG](inputs/ffeac91953b3eee38861b2d393ef7c3fc858ddadf476dde1234e1291378650e5.png) |
| training | 21 | 0 | text | "唐代" | "唐代" | True | native text |
| training | 21 | 1 | visual | "代" | "代" | True | [PNG](inputs/d62fd2026ab575265ba74d102e23e19644a30070eb85df7c99cf71457333be2a.png) |
| training | 21 | 1 | text | "代" | "代" | True | native text |
| training | 21 | 1 | erased | "代" | "唐代" | False | [PNG](inputs/ffeac91953b3eee38861b2d393ef7c3fc858ddadf476dde1234e1291378650e5.png) |
| training | 21 | 2 | visual | "" | "唐代" | False | [PNG](inputs/b5ac67db41459bef54ddc29b7cfbe6370bd78411857b06a283078cf812041902.png) |
| training | 21 | 2 | text | "" | "始于唐代" | False | native text |
| training | 22 | 0 | visual | "阿留申黑颊黑雁或阿留申鹅" | "阿留申黑颊黑雁或阿留申鹅" | True | [PNG](inputs/686df275f7ec7b4688ec72bdf22133e50a50693f7d7c4f53f9ec8b4acff4fa05.png) |
| training | 22 | 0 | text | "阿留申黑颊黑雁或阿留申鹅" | "阿留申黑颊黑雁或阿留申鹅" | True | native text |
| training | 22 | 6 | visual | "雁或阿留申鹅" | "雁" | False | [PNG](inputs/f366bf24b0784e3aa560f3b7984959caf5ea8695478c88ae7c6801d52f355e40.png) |
| training | 22 | 6 | text | "雁或阿留申鹅" | "雁或阿留申鹅" | True | native text |
| training | 22 | 6 | erased | "雁或阿留申鹅" | "阿留申黑颊黑雁或阿留申鹅" | False | [PNG](inputs/686df275f7ec7b4688ec72bdf22133e50a50693f7d7c4f53f9ec8b4acff4fa05.png) |
| training | 22 | 11 | visual | "鹅" | "鹅" | True | [PNG](inputs/52f303ec7d517e2f22791ad3bac700935df80eb7571073f996302d1bd2779c93.png) |
| training | 22 | 11 | text | "鹅" | "鹅" | True | native text |
| training | 22 | 12 | visual | "" | "阿留申黑颊黑雁或阿留申鹅" | False | [PNG](inputs/8570db42cc2ff253b65e6a73931bab2510cb1d4ef7c239d74f704ab13de8ad3a.png) |
| training | 22 | 12 | text | "" | "" | True | native text |
| training | 23 | 0 | visual | "1.7-2.1公斤" | "1.7-2.1公斤" | True | [PNG](inputs/955e990e5419aaea2617d1ef1a84f8692b3e3ae4a79b3b106fa840eb21ec1ccf.png) |
| training | 23 | 0 | text | "1.7-2.1公斤" | "1.7-2.1公斤" | True | native text |
| training | 23 | 4 | visual | "2.1公斤" | "2.1公斤" | True | [PNG](inputs/d5d95a886487c1c69007b7efef6500457ed7477e374ce2c77e8e0e4e11c88bb1.png) |
| training | 23 | 4 | text | "2.1公斤" | "2.1公斤" | True | native text |
| training | 23 | 4 | erased | "2.1公斤" | "1.7-2.1公斤" | False | [PNG](inputs/955e990e5419aaea2617d1ef1a84f8692b3e3ae4a79b3b106fa840eb21ec1ccf.png) |
| training | 23 | 8 | visual | "斤" | "公斤" | False | [PNG](inputs/f17534806e497d5b909f61a8d6ae15d3d58ca34b2e838ba32791a4040ed0ffe6.png) |
| training | 23 | 8 | text | "斤" | "斤" | True | native text |
| training | 23 | 9 | visual | "" | "1.7-2.1公斤" | False | [PNG](inputs/51f2c464a0f13806e792effbff9d3a1dde643708c6b629db231df8559d45ad39.png) |
| training | 23 | 9 | text | "" | "" | True | native text |
| training | 24 | 0 | visual | "总部设于法国布洛涅-比扬古" | "法国布洛涅-比扬古" | False | [PNG](inputs/6a38879f426166e873026bf8e6cf2ee1b056b6544d7f33bc895f3a0e7fb23b50.png) |
| training | 24 | 0 | text | "总部设于法国布洛涅-比扬古" | "法国布洛涅-比扬古" | False | native text |
| training | 24 | 6 | visual | "布洛涅-比扬古" | "布洛涅-比扬古" | True | [PNG](inputs/604fd7faaf6e961f50ae9768225c5642739dbe1ec9e0cd22cb45c4c06e37f32e.png) |
| training | 24 | 6 | text | "布洛涅-比扬古" | "布洛涅-比扬古" | True | native text |
| training | 24 | 6 | erased | "布洛涅-比扬古" | "法国布洛涅-比扬古" | False | [PNG](inputs/6a38879f426166e873026bf8e6cf2ee1b056b6544d7f33bc895f3a0e7fb23b50.png) |
| training | 24 | 12 | visual | "古" | "-比扬古" | False | [PNG](inputs/f82d2d428c663bc68794045177277f1f6a6763c8af36d50e0cfcd9350f865ad2.png) |
| training | 24 | 12 | text | "古" | "古" | True | native text |
| training | 24 | 13 | visual | "" | "法国布洛涅-比扬古" | False | [PNG](inputs/212347553dd8f1bab5665a774e5bf3cfb958cc22bec430316cee6826f3d4f5a6.png) |
| training | 24 | 13 | text | "" | "法国布洛涅-比扬古" | False | native text |
| training | 25 | 0 | visual | "阿尔卡特成立于1898年" | "1898年" | False | [PNG](inputs/b1d2731bd80d126f1e75e4f72b557daca672954cd5dd0813b8f4a9094960444f.png) |
| training | 25 | 0 | text | "阿尔卡特成立于1898年" | "2006" | False | native text |
| training | 25 | 6 | visual | "于1898年" | "于1898年" | True | [PNG](inputs/e1f31860de3ae16f853e2360b851024f68a0259082903dded3575fd6cf3736e8.png) |
| training | 25 | 6 | text | "于1898年" | "于1898年" | True | native text |
| training | 25 | 6 | erased | "于1898年" | "1898年" | False | [PNG](inputs/b1d2731bd80d126f1e75e4f72b557daca672954cd5dd0813b8f4a9094960444f.png) |
| training | 25 | 11 | visual | "年" | "1898" | False | [PNG](inputs/c2dea911d638b68f2e660926ee730aebd2975f06958dbfe02f941c93962dad22.png) |
| training | 25 | 11 | text | "年" | "年，朗讯科技成立于1996年，两者于2006年合并成立阿尔卡特-朗讯。" | False | native text |
| training | 25 | 12 | visual | "" | "阿尔卡特-朗讯成立于2006年" | False | [PNG](inputs/ead2742b1b4260b845d4eb86b6889a3011517bdd7c5c4bbc13215d38042f47aa.png) |
| training | 25 | 12 | text | "" | "2006年" | False | native text |
| training | 26 | 0 | visual | "慕尼黑市中心玛利亚广场东侧" | "位于慕尼黑市中心玛利亚广场东侧" | False | [PNG](inputs/5aacf8949162fc98d67879b6aeef2a0386e8d0d18b8f11ea182098ada101c174.png) |
| training | 26 | 0 | text | "慕尼黑市中心玛利亚广场东侧" | "慕尼黑市中心玛利亚广场东侧" | True | native text |
| training | 26 | 6 | visual | "玛利亚广场东侧" | "玛利亚广场东侧" | True | [PNG](inputs/eef6c3077c4f7568a397129e67a2ba282a53ea96533f2e4fa70aac607d65331a.png) |
| training | 26 | 6 | text | "玛利亚广场东侧" | "玛利亚广场东侧" | True | native text |
| training | 26 | 6 | erased | "玛利亚广场东侧" | "位于慕尼黑市中心玛利亚广场东侧" | False | [PNG](inputs/5aacf8949162fc98d67879b6aeef2a0386e8d0d18b8f11ea182098ada101c174.png) |
| training | 26 | 12 | visual | "侧" | "东" | False | [PNG](inputs/a8adcbd3d10a2e69c2aa8e7108fee2264b69781e9c03c6dd698219638d3835c4.png) |
| training | 26 | 12 | text | "侧" | "侧" | True | native text |
| training | 26 | 13 | visual | "" | "慕尼黑市中心玛利亚广场东侧" | False | [PNG](inputs/ac9cdd877cd4a8bc8e7ead73fd4121ecf6c2a10f9e0921ae29e6fa303ddc6f95.png) |
| training | 26 | 13 | text | "" | "现在供市议会代表使用" | False | native text |
| training | 27 | 0 | visual | "1310年" | "1310年" | True | [PNG](inputs/86c93c8f62243e9ba9c68ae3958c3e089e014138f983aec2f184c6946285934d.png) |
| training | 27 | 0 | text | "1310年" | "1310年" | True | native text |
| training | 27 | 2 | visual | "10年" | "1310年" | False | [PNG](inputs/58cbe731f921b4de4664b9fbfa5d0c043cdb0e8d77e7133c6b92ff29fbff5ba1.png) |
| training | 27 | 2 | text | "10年" | "10年" | True | native text |
| training | 27 | 2 | erased | "10年" | "1310年" | False | [PNG](inputs/86c93c8f62243e9ba9c68ae3958c3e089e014138f983aec2f184c6946285934d.png) |
| training | 27 | 4 | visual | "年" | "1310" | False | [PNG](inputs/a58f23a8634ba93720a2a5f784ffad383196b72f1730e04f767c072654cb5938.png) |
| training | 27 | 4 | text | "年" | "年" | True | native text |
| training | 27 | 5 | visual | "" | "1310年" | False | [PNG](inputs/d186db5354a9d42e5a38cf90ad106bed32a26d62387d44736df6c414a862dd94.png) |
| training | 27 | 5 | text | "" | "" | True | native text |
| training | 28 | 0 | visual | "美国著名的电子游戏软件制作商" | "美国著名的电子游戏软件制作商" | True | [PNG](inputs/c151c48891ea60240949105900eae7cbaacc7e1b5feb0ce21d6dd77d3af6d2e8.png) |
| training | 28 | 0 | text | "美国著名的电子游戏软件制作商" | "美国著名的电子游戏软件制作商" | True | native text |
| training | 28 | 7 | visual | "游戏软件制作商" | "游戏软件制作商" | True | [PNG](inputs/c022c47691b030901b8facf90f4b57dfb74a152741fdca0f1e885030ef8f007d.png) |
| training | 28 | 7 | text | "游戏软件制作商" | "游戏软件制作商" | True | native text |
| training | 28 | 7 | erased | "游戏软件制作商" | "美国著名的电子游戏软件制作商" | False | [PNG](inputs/c151c48891ea60240949105900eae7cbaacc7e1b5feb0ce21d6dd77d3af6d2e8.png) |
| training | 28 | 13 | visual | "商" | "商" | True | [PNG](inputs/e957997b75d4da3d00a201aafe1df365f5380ccdd87d8ee7e48619816830b486.png) |
| training | 28 | 13 | text | "商" | "商，是由两名芝加哥大学的本科生AlexSeropian和JasonJones于1991年5月以BungieSoftwareProductsCorporation的名称成立。公司最初设立在芝加哥，在成立的最初九年里，公司主要致力于Macintosh平台游戏的开发，他们开发了深受玩家喜爱的" | False | native text |
| training | 28 | 14 | visual | "" | "美国著名的电子游戏软件制作商" | False | [PNG](inputs/ec4c1202779fac376052ce2cf652048e2bc6876a480ca6beb4b3efe1de77622e.png) |
| training | 28 | 14 | text | "" | "是由两名芝加哥大学的本科生AlexSeropian和JasonJones于1991年5月以BungieSoftwareProductsCorporation的名称成立" | False | native text |
| training | 29 | 0 | visual | "1991年" | "1991年" | True | [PNG](inputs/edb7b2658b4286dffa86cf43dd6507b1bd4087ba02cb330741acc6db735e9d54.png) |
| training | 29 | 0 | text | "1991年" | "1991年" | True | native text |
| training | 29 | 2 | visual | "91年" | "91" | False | [PNG](inputs/a0f428c443936bb9b1536b04f6950ced3ae1f7c8b6bd3bbc565ef9572197eea5.png) |
| training | 29 | 2 | text | "91年" | "91" | False | native text |
| training | 29 | 2 | erased | "91年" | "1991年" | False | [PNG](inputs/edb7b2658b4286dffa86cf43dd6507b1bd4087ba02cb330741acc6db735e9d54.png) |
| training | 29 | 4 | visual | "年" | "1991" | False | [PNG](inputs/4b586cef648e4e9d4ef89551a617bbe2af2bee57335819010c0916d46ab89250.png) |
| training | 29 | 4 | text | "年" | "年5月" | False | native text |
| training | 29 | 5 | visual | "" | "1991年" | False | [PNG](inputs/b8ca36c1cbb847eaba0aee608c7ca97fdda221d98e33ba1cc5e4382b61dae039.png) |
| training | 29 | 5 | text | "" | "5月" | False | native text |
| training | 30 | 0 | visual | "澳门巴士2A路线" | "2A路线" | False | [PNG](inputs/7b511a4969c359723164a009f0b85d7ad6c59a4615d0266beac838db36b53175.png) |
| training | 30 | 0 | text | "澳门巴士2A路线" | "澳门巴士2A路线" | True | native text |
| training | 30 | 4 | visual | "2A路线" | "2A路线" | True | [PNG](inputs/44eb809c7b4a3f11c01462a492f834d1c7891809c5ab06f2a4f82198231bb8bb.png) |
| training | 30 | 4 | text | "2A路线" | "2A路线" | True | native text |
| training | 30 | 4 | erased | "2A路线" | "2A路线" | True | [PNG](inputs/7b511a4969c359723164a009f0b85d7ad6c59a4615d0266beac838db36b53175.png) |
| training | 30 | 7 | visual | "线" | "澳门巴士2A路" | False | [PNG](inputs/1a2b052b21ceea655ccd65328361ea1e8f739f92371052c76cbe717af64086d2.png) |
| training | 30 | 7 | text | "线" | "线" | True | native text |
| training | 30 | 8 | visual | "" | "澳门巴士2A路线" | False | [PNG](inputs/ad6d1989151089a4c663a735a85519536049b78f53ecc8410fe6629ccfc444da.png) |
| training | 30 | 8 | text | "" | "" | True | native text |
| training | 31 | 0 | visual | "本线亦没有任何\"独市位\"" | "没有" | False | [PNG](inputs/37042411405fc261545094b54a56c6fb2aed24742cd553bdad0bc26242e99a26.png) |
| training | 31 | 0 | text | "本线亦没有任何\"独市位\"" | "没有" | False | native text |
| training | 31 | 6 | visual | "何\"独市位\"" | "任何\"独市位\"" | False | [PNG](inputs/0eb29cbe4270b4f70196f443bd7b129376438cd4155a3f2b5a2f9c47bc1518db.png) |
| training | 31 | 6 | text | "何\"独市位\"" | "何“独市位”" | False | native text |
| training | 31 | 6 | erased | "何\"独市位\"" | "没有" | False | [PNG](inputs/37042411405fc261545094b54a56c6fb2aed24742cd553bdad0bc26242e99a26.png) |
| training | 31 | 11 | visual | "\"" | "\"独市位\"" | False | [PNG](inputs/e8e2444d70f1a9116569b67b15f173e5ee2def8e1d23358961e30005e9e97bd5.png) |
| training | 31 | 11 | text | "\"" | "，令客量十分低迷" | False | native text |
| training | 31 | 12 | visual | "" | "本线亦没有任何\"独市位\"" | False | [PNG](inputs/07a7c7013526fb5e3598a06829ae0b95c7db40a38a6239a004ee5a56978854a5.png) |
| training | 31 | 12 | text | "" | "令客量十分低迷" | False | native text |
