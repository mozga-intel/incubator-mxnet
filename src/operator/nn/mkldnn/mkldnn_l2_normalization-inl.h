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
 * \file mkldnn_l2_normalization-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_MKLDNN_L2_NORMALIZATION_INL_H_
#define MXNET_OPERATOR_MKLDNN_L2_NORMALIZATION_INL_H_

//#if MXNET_USE_MKLDNN == 1

#include <mxnet/operator.h>
#include <iostream>
#include <mkldnn.hpp>
#include "./mkldnn_base-inl.h"

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "../../mshadow_op.h"
#include "../../operator_common.h"

#ifdef _MSC_VER
#define collapse(x)
#endif

namespace mxnet {
namespace op {

template <typename cpu, typename TParam, typename DType>
class MKLDNNL2_NormalizationOpCPU : public Operator {
 public:
  explicit MKLDNNL2_NormalizationOpCPU(TParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    const int device_index = in_data[0].dev_id(); 
    const int device_index1 = in_data[0].dev_id();

    NDArray input_data(in_data[0], device_index);
    NDArray output_data(out_data[0], device_index);

    mxnet::TShape orig_shape = in_data[0].shape_;
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    auto datas = input_data.data();

    auto omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
if (this->param_.mode == 0) {
    mshadow::Shape<2> dshape =
        mshadow::Shape2(orig_shape[0],
                        orig_shape.ProdShape(1, orig_shape.ndim()));
      mshadow::Shape<1>  norm_shape1 = mshadow::Shape1(dshape[0]);
      mshadow::Tensor<cpu, 1, DType> norm1 =
            out_data[1].get_with_shape<cpu, 1, DType>(norm_shape1, s);

      mshadow::Tensor<cpu, 2, DType> data1 = in_data[0]
        .get_with_shape<cpu, 2, DType>(dshape, s);

      mshadow::Shape<2> norm_shape =
        mshadow::Shape2(dshape[0], dshape[1] );
      mshadow::Tensor<cpu, 2, DType> norm =
        out_data[0].get_with_shape<cpu,2, DType>(norm_shape, s);
#pragma omp parallel for num_threads(omp_threads)
      for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
        norm1[shape0] = DType(this->param_.eps);
        for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
          norm1[shape0] += data1[shape0][shape1] * data1[shape0][shape1];
        }
        for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
          norm[shape0][shape1] = 1.0 / std::sqrt(norm1[shape0]);
        }
      }
    NDArray norm_data(norm, device_index);
    NDArray data(data1, device_index);
    Compute(data, norm_data, output_data, req[0], ctx, out_data);
} else {
   mshadow::Shape<3> dshape =
        mshadow::Shape3(orig_shape[0], orig_shape[1],
                        orig_shape.ProdShape(2, orig_shape.ndim()));
    mshadow::Tensor<cpu, 3, DType> data1 =
        datas.get_with_shape<cpu, 3, DType>(dshape, s);

    mshadow::Shape<3> norm_shape =
        mshadow::Shape3(dshape[0], dshape[1], dshape[2]);
    mshadow::Tensor<cpu, 3, DType> norm =
        out_data[0].get_with_shape<cpu, 3, DType>(norm_shape, s);
        
    if (this->param_.mode == 1) {
      mshadow::Shape<2> norm_shape1 = mshadow::Shape2(dshape[0], dshape[2]);
      mshadow::Tensor<cpu, 2, DType> norm1 =
            out_data[1].get_with_shape<cpu, 2, DType>(norm_shape1, s);
      #pragma omp parallel for num_threads(omp_threads) collapse(2)
      for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
          for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
            norm1[shape0][shape2] = DType(this->param_.eps);
            for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
              norm1[shape0][shape2] +=
                  data1[shape0][shape1][shape2] * data1[shape0][shape1][shape2];
            }
            for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
              norm[shape0][shape1][shape2] = 1.0 / std::sqrt(norm1[shape0][shape2]);
            }
          }
        }
    } else if(this->param_.mode == 2) {
      mshadow::Shape<2> norm_shape1 = mshadow::Shape2(dshape[0], dshape[1]);
      mshadow::Tensor<cpu, 2, DType> norm1 =
      out_data[1].get_with_shape<cpu, 2, DType>(norm_shape1, s);
      #pragma omp parallel for num_threads(omp_threads) collapse(2)
          for (int shape0 = 0; shape0 < static_cast<int>(dshape[0]); shape0++) {
            for (int shape1 = 0; shape1 < static_cast<int>(dshape[1]); shape1++) {
              norm1[shape0][shape1] = DType(this->param_.eps);
              for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
                norm1[shape0][shape1] += data1[shape0][shape1][shape2] * data1[shape0][shape1][shape2];
              }
              for (int shape2 = 0; shape2 < static_cast<int>(dshape[2]); shape2++) {
                norm[shape0][shape1][shape2] = 1.0 / std::sqrt(norm1[shape0][shape1]);
              }
            }
          }
    }
      NDArray norm_data(norm, device_index);
      NDArray data(data1, device_index);
      Compute(data, norm_data, output_data, req[0], ctx, out_data);
  }
  }

 public:
  TParam param_;
  bool with_workspace_;
  std::shared_ptr<mkldnn::binary::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::binary> fwd_;

 public:
  void Init(const mxnet::NDArray &input, const mxnet::NDArray &norm,
            const mxnet::NDArray &output, const mkldnn::algorithm alg_kind,
            const bool with_workspace, const OpContext &ctx,
            const std::vector<TBlob> &in_data) {
    //auto dimension =  {input.shape()[0], input.shape()[1], input.shape()[2]};
    //auto format_tag = mkldnn::memory::format_tag::abc;
    //if(input.shape().ndim() == 2) {
     auto dimension =  {input.shape()[0], input.shape()[1]};
     auto  format_tag = mkldnn::memory::format_tag::ab;
    //}
    const auto src_md = mkldnn::memory::desc(
        dimension, mkldnn::memory::data_type::f32, format_tag);
    const auto norm_md = mkldnn::memory::desc(
       dimension,
        mkldnn::memory::data_type::f32, format_tag);
    const auto dst_md = GetMemDesc(output);
    const mkldnn::engine engine = CpuEngine::Get()->get_engine();
    const auto fwd_desc = mkldnn::binary::desc(mkldnn::algorithm::binary_mul,
                                               src_md, norm_md, dst_md);
    this->fwd_pd_.reset(new mkldnn::binary::primitive_desc(fwd_desc, engine));
    this->fwd_.reset(new mkldnn::binary(*(this->fwd_pd_)));
  }

  void Execute(const mxnet::NDArray &input, const mxnet::NDArray &norm,
               const mxnet::NDArray &output, const OpReqType req,
               const OpContext &ctx, const std::vector<TBlob> &in_data) {
    NDArray in_buffer = input;
    auto input_mem = input.GetMKLDNNData();
    auto norm_md = GetWeights(norm, this->fwd_pd_->src1_desc(), 0);
    auto output_mem_t = CreateMKLDNNMem(output, this->fwd_pd_->dst_desc(), req);

    mkldnn_args_map_t args = {{MKLDNN_ARG_SRC_0, *input_mem},
                              {MKLDNN_ARG_SRC_1, *norm_md},
                              {MKLDNN_ARG_DST, *(output_mem_t.second)}};
    if (this->fwd_) {
      MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
      CommitOutput(output, output_mem_t);
      MKLDNNStream::Get()->Submit();
    } else {
      LOG(FATAL) << "OneDNN L2 Norm primitive is nullptr";
    }
  }

  MKLDNNL2_NormalizationOpCPU(const mxnet::NDArray &input,
                              const mxnet::NDArray &norm,
                              const mxnet::NDArray &output,
                              const mkldnn::algorithm alg_kind,
                              const bool with_workspace, const OpContext &ctx,
                              const std::vector<TBlob> &in_data)
      : with_workspace_(with_workspace), fwd_pd_(nullptr) {
    Init(input, norm, output, alg_kind, with_workspace, ctx, in_data);
  }

  MKLDNNL2_NormalizationOpCPU &GetL2Fwd(const mxnet::NDArray &input,
                                        const mxnet::NDArray &norm,
                                        const mxnet::NDArray &output,
                                        const OpContext &ctx,
                                        const std::vector<TBlob> &in_data) {
    typedef ParamOpSign<TParam> MKLDNNL2Signature;
#if DMLC_CXX11_THREAD_LOCAL
    static thread_local std::unordered_map<MKLDNNL2Signature,
                                           MKLDNNL2_NormalizationOpCPU, OpHash>
        l2_fwds;
#else
    static MX_THREAD_LOCAL std::unordered_map<
        MKLDNNL2Signature, MKLDNNL2_NormalizationOpCPU, OpHash>
        l2_fwds;
#endif
    bool with_workspace = true;
    MKLDNNL2Signature key(param_);
    key.AddSign(with_workspace);
    key.AddSign(input);
    key.AddSign(output);
    auto it = l2_fwds.find(key);
    if (it == l2_fwds.end()) {
      auto data_md = input.GetMKLDNNData()->get_desc();
      mkldnn::algorithm kind = mkldnn::algorithm::reduction_norm_lp_power_p_sum;
      MKLDNNL2_NormalizationOpCPU fwd(input, norm, output, kind, false, ctx,
                                      in_data);
      it = AddToCache(&l2_fwds, key, fwd);
    }
    return it->second;
  }

  void Compute(const mxnet::NDArray &input, const mxnet::NDArray &norm,
               const mxnet::NDArray &output, const OpReqType req,
               const OpContext &ctx, const std::vector<TBlob> &in_data) {
    auto &fwd = GetL2Fwd(input, norm, output, ctx, in_data);
    fwd.Execute(input, norm, output, req, ctx, in_data);
  }
};

}  // namespace op
}  // namespace mxnet

//#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_MKLDNN_L2_NORMALIZATION_INL_H_