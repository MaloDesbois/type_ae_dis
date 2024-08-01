def masking(datas, mask,p) :
  mask_b = torch.where(mask==1, torch.bernoulli(1-p),mask)
  indices = torch.where(mask!=mask_b)
  datas_b = datas.copy()
  datas_b[indices] = 0
  return (datas_b, mask_b, indices)
