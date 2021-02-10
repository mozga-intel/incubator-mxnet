/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file mkldnn_adaptive_pooling-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_ADAPTIVE_POOLING_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_ADAPTIVE_POOLING_INL_H_

#if MXNET_USE_MKLDNN == 1

#include "../../operator_common.h"
#include "./mkldnn_base-inl.h"
#include <mkldnn.hpp>
#include <utility>

namespace mxnet {
namespace op {

class MKLDNNAdaptivePoolingFwd {
public:
  MKLDNNAdaptivePoolingFwd(const mxnet::NDArray &input,
                           const mxnet::NDArray &output,
                           const mkldnn::memory::dims &kernel,
                           const mkldnn::memory::dims &strides,
                           const mkldnn::memory::dims &pad_l,
                           const mkldnn::memory::dims &pad_r,
                           const mkldnn::algorithm alg_kind,
                           const bool with_workspace, const bool is_train)
      : with_workspace_(with_workspace), fwd_(nullptr) {
    Init(input, output, kernel, strides, pad_l, pad_r, is_train, alg_kind);
  }
  ~MKLDNNAdaptivePoolingFwd() = default;

public:
  void Execute(const NDArray &input, const OpReqType req, const NDArray &output,
               const NDArray *workspace);

private:
  bool with_workspace_;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::pooling_forward> fwd_;

private:
  void Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
            const mkldnn::memory::dims &kernel,
            const mkldnn::memory::dims &strides,
            const mkldnn::memory::dims &pad_l,
            const mkldnn::memory::dims &pad_r, const bool is_train,
            const mkldnn::algorithm alg_kind);
};

class MKLDNNAdaptivePoolingBwd {
public:
  MKLDNNAdaptivePoolingBwd(
      const mkldnn::pooling_backward::primitive_desc &pdesc, bool with_ws)
      : with_workspace(with_ws), pd(pdesc) {
    bwd = std::make_shared<mkldnn::pooling_backward>(pd);
  }
  ~MKLDNNAdaptivePoolingBwd() = default;

public:
  const mkldnn::pooling_backward::primitive_desc pd;
  const mkldnn::pooling_backward &GetBwd();
  const mkldnn::pooling_backward::primitive_desc &GetPd();

private:
  std::shared_ptr<const mkldnn::pooling_backward> bwd;
  bool with_workspace;
};

template <typename T = mkldnn::memory::dims>
void updateAdaptivePaddingKernel(T &kernel, T &strides, T &pad_l, T &pad_r,
                                 const NDArray &in_data,
                                 const NDArray &out_data) {
  const int IH = in_data.shape()[2];
  const int IW = in_data.shape()[3];
  const int OH = out_data.shape()[2];
  const int OW = out_data.shape()[3];

  strides.at(0) = floor((IH << 1) / OH) - floor(IH / OH);
  strides.at(1) = floor((IW << 1) / OW) - floor(IW / OW);
  kernel.at(0) = ceil((IH << 1) / OH) - floor(IH / OH);
  kernel.at(1) = ceil((IW << 1) / OW) - floor(IW / OW);
  pad_l.at(0) = (strides.at(0) * (OH - 1) + kernel.at(0) - IH) >> 1;
  pad_l.at(1) = (strides.at(1) * (OW - 1) + kernel.at(1) - IW) >> 1;
}

template <typename T>
MKLDNNAdaptivePoolingFwd &GetPoolingFwd(const T &param, const bool is_train,
                                        const NDArray &input,
                                        const NDArray &output) {
  if (input.shape().ndim() != 4) {
    LOG(FATAL) << "MKLDNN Adaptive Avg Pool 2d: Expect only 2D input";
  }
  typedef ParamOpSign<T> MKLDNNPoolingSignature;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNAdaptivePoolingFwd, OpHash>
      pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNAdaptivePoolingFwd, OpHash>
      pooling_fwds;
#endif
  bool with_workspace = is_train && true;
  MKLDNNPoolingSignature key(param);
  key.AddSign(is_train);
  key.AddSign(with_workspace);
  key.AddSign(input);
  key.AddSign(output);

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    auto data_md = input.GetMKLDNNData()->get_desc();
    const int kernel_ndims = input.shape().ndim();

    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    mkldnn::memory::dims pad_l(kernel_ndims);
    mkldnn::memory::dims pad_r(kernel_ndims);

    updateAdaptivePaddingKernel(kernel, strides, pad_l, pad_r, input, output);
    mkldnn::memory::validate_dims(kernel);
    mkldnn::memory::validate_dims(strides);
    mkldnn::memory::validate_dims(pad_l);
    mkldnn::memory::validate_dims(pad_r);

    mkldnn::algorithm kind = mkldnn::algorithm::pooling_avg;
    MKLDNNAdaptivePoolingFwd fwd(input, output, kernel, kernel, pad_l, pad_r,
                                 kind, false, false);
    it = AddToCache(&pooling_fwds, key, fwd);
  }
  return it->second;
}

static inline mkldnn::memory::data_type
get_data_type(const mkldnn::memory::desc &md) {
  return static_cast<mkldnn::memory::data_type>(md.data_type());
}

template <typename Param>
mkldnn::pooling_forward::primitive_desc
GetAdaptivePoolingFwdPdesc(const Param &param, const bool is_train,
                           const NDArray &input, const NDArray &output) {

  const auto src_md = input.GetMKLDNNData()->get_desc();
  const auto dst_md = GetMemDesc(output);

  const int kernel_ndims = input.shape().ndim();
  mkldnn::memory::dims kernel(kernel_ndims - 2);
  mkldnn::memory::dims strides(kernel_ndims - 2);
  mkldnn::memory::dims pad_l(kernel_ndims - 2);
  mkldnn::memory::dims pad_r(kernel_ndims - 2);

  updateAdaptivePaddingKernel(kernel, strides, pad_l, pad_r, input, output);

  const mkldnn::algorithm alg = mkldnn::algorithm::pooling_avg;
  mkldnn::prop_kind kind = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg != mkldnn::algorithm::pooling_avg) {
    kind = mkldnn::prop_kind::forward_training;
  }

  const mkldnn::pooling_forward::desc poolingFwd_desc(
      kind, alg, src_md, dst_md, strides, kernel, pad_l, pad_r);
  return mkldnn::pooling_forward::primitive_desc(
      poolingFwd_desc, CpuEngine::Get()->get_engine());
}

template <typename T>
MKLDNNAdaptivePoolingBwd &GetPoolingBwd(const T &param, const NDArray &in_data,
                                        const NDArray &in_grad,
                                        const NDArray &out_grad) {
  typedef ParamOpSign<T> MKLDNNPoolingSignature;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNAdaptivePoolingBwd, OpHash>
      pooling_bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNAdaptivePoolingBwd, OpHash>
      pooling_bwds;
#endif
  bool with_workspace = MKLDNNRequireWorkspace(param);
  MKLDNNPoolingSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);

  auto it = pooling_bwds.find(key);
  if (it == pooling_bwds.end()) {
    auto input_mem = in_data.GetMKLDNNData();
    const mkldnn::memory::desc data_md = input_mem->get_desc();

    auto dst_dims =
        mkldnn::memory::dims(out_grad.shape().begin(), out_grad.shape().end());
    auto any = mkldnn::memory::format_tag::any;
    auto dst_md = mkldnn::memory::desc(dst_dims, get_data_type(data_md), any);

    auto fwd_pd = GetAdaptivePoolingFwdPdesc(param, true, in_data, out_grad);

    auto diff_src_dims =
        mkldnn::memory::dims(in_grad.shape().begin(), in_grad.shape().end());
    auto diff_src_md =
        mkldnn::memory::desc(diff_src_dims, get_data_type(data_md), any);
    auto cpu_engine = CpuEngine::Get()->get_engine();
    auto alg = mkldnn::algorithm::pooling_avg_include_padding;

    const int kernel_ndims = in_grad.shape().ndim();
    mkldnn::memory::dims kernel(kernel_ndims - 2);
    mkldnn::memory::dims strides(kernel_ndims - 2);
    mkldnn::memory::dims pad_l(kernel_ndims - 2);
    mkldnn::memory::dims pad_r(kernel_ndims - 2);

    updateAdaptivePaddingKernel(kernel, strides, pad_l, pad_r, in_grad,
                                out_grad);

    auto bwd_desc = mkldnn::pooling_backward::desc(
        alg, diff_src_md, dst_md, strides, kernel, pad_l, pad_r);
    auto pdesc =
        mkldnn::pooling_backward::primitive_desc(bwd_desc, cpu_engine, fwd_pd);

    MKLDNNAdaptivePoolingBwd bwd(pdesc, with_workspace);
    it = AddToCache(&pooling_bwds, key, bwd);
  }
  return it->second;
}

template <typename T>
void MKLDNNAdaptivePoolingCompute(const OpContext &ctx, const T &param,
                                  const NDArray &in_data, const OpReqType req,
                                  const NDArray &out_data,
                                  const NDArray *workspace) {
  auto &fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
  fwd.Execute(in_data, req, out_data, workspace);
}

template <typename T>
void MKLDNNAdaptivePoolingGradCompute(const OpContext &ctx, const T &param,
                                      const NDArray &out_grad,
                                      const NDArray &in_data,
                                      const NDArray *workspace,
                                      const OpReqType req,
                                      const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }

  TmpMemMgr::Get()->Init(ctx.requested[0]);

  auto &bwd = GetPoolingBwd(param, in_data, in_grad, out_grad);
  auto diff_dst_mem = out_grad.GetMKLDNNDataReorder(bwd.pd.diff_dst_desc());
  auto diff_src_mem = CreateMKLDNNMem(in_grad, bwd.pd.diff_src_desc(), req);

  mkldnn_args_map_t args = {
      {MKLDNN_ARG_DIFF_DST, *diff_dst_mem},
      {MKLDNN_ARG_DIFF_SRC, *diff_src_mem.second},
  };

  if (MKLDNNRequireWorkspace(param) && workspace != nullptr) {
    args[MKLDNN_ARG_WORKSPACE] = *(workspace->GetMKLDNNData());
  }

  MKLDNNStream::Get()->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Get()->Submit();
}
} // namespace op
} // namespace mxnet
#endif // MXNET_USE_MKLDNN == 1
#endif // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_
