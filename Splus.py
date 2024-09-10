from Preprocess import Preprocess, Memristor, Memr_init
from Trainmodel import Trainmodel, Memr_Weigt_range
from Net import Net
import argparse
import torch
import torch.nn as nn

torch.manual_seed(125)
para_all=argparse.ArgumentParser()
para_all.add_argument("--batch_size",type=int,default=600)
para_all.add_argument("--learning_rate_sware",type=float,default=0.8)
para_all.add_argument("--learning_rate",type=float,default=0.6)
para_all.add_argument("--epoches_sware",type=int,default=1)
para_all.add_argument("--epoches",type=int,default=1)
para_all.add_argument("--v",type=int,default=2)
para_all.add_argument("--pnum",type=int,default=40)
para_all.add_argument("--threadscale",type=float,default=1)
para_all.add_argument("--onoff_ratio",type=float,default=1)
para_all.add_argument("--noise_rate",type=float,default=0)
# para_all.add_argument("--save_model_path",type=str,default="./")
parg=para_all.parse_args()


pp = Preprocess(parg.batch_size)
pp.loaddate()


model_sware = Net(28 * 28, 10)
# if torch.cuda.is_available():
#     model_sware = model_sware.cuda()
torch.nn.init.constant_(model_sware.fc1.weight,0)
tt = Trainmodel(pp, parg.epoches_sware, parg.batch_size, parg.learning_rate_sware, model_sware, ON_OFF=False)
# tt.train()

memr = Memristor(parg.v, parg.pnum,parg.onoff_ratio)
memr.prodmemr()
memr.memrtreat()
#nomorlize noise
noise=(memr.G_end-memr.G_beg)*parg.noise_rate
mwr = Memr_Weigt_range(model_sware, memr.G_scale, parg.pnum,parg.threadscale)
mwr.m_Wrange()
print("W_beg:{:.3f}..".format(mwr.W_beg),
      "W_end:{:.3f}..".format(mwr.W_end),
      "Wg_scale:{:.3f}..".format(mwr.wg_scale),
      "splus_weight:{:.3f}..".format(mwr.sp_w))

model = Net(28 * 28, 10)
torch.nn.init.constant_(model.fc1.weight,mwr.W_beg)


mminit = Memr_init(model, mwr.wg_scale, memr.Gplus)
mminit.mrmr_init()
# for p in model.parameters():
#     p.data=mminit.pdata*(-1)
# temp_state = model.state_dict()
# input_weight = temp_state['fc1.weight']

ttm = Trainmodel(pp, parg.epoches, parg.batch_size, parg.learning_rate, model, ON_OFF=True, sp_w=mwr.sp_w, plusarr=mminit.plusarr,
                 wg_scale=mwr.wg_scale, Gplus=memr.Gplus,LTP_Gplus=memr.LTP_Gplus,LTD_Gplus=memr.LTD_Gplus,pnum=parg.pnum,
                 v=parg.v,threadscale=parg.threadscale,onoff_ratio=parg.onoff_ratio,noise_rate=parg.noise_rate,noise=noise)
ttm.train()
print("a")
print("b")
print("d")
print("c")
print("success")
print("test_dhy")