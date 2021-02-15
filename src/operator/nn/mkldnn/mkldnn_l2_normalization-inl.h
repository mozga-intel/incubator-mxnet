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

#include <mkldnn.hpp>
#include <mxnet/operator.h>
namespace mxnet {
namespace op {

template <typename T> struct Is {
  const T d_;
  template <class... Args> bool in(Args... args) {
    bool r{false};
    [](...) {}(((r = r || d_ == args), 1)...);
    return r;
  }
};

template <class T> Is<T> is(T d) { return Is<T>{d}; }

template <typename cpu, typename TParam>
class MKLDNNL2_NormalizationOpCPU : public Operator {
public:
  explicit MKLDNNL2_NormalizationOpCPU(TParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 2U);
    const int device_index = out_data[0].dev_id();
    NDArray input_data(in_data[0], device_index);
    NDArray output_data1(out_data[1], device_index);
    Compute(input_data, output_data1, req[0]);
  }

protected:
  TParam param_;
  bool with_workspace_;
  std::shared_ptr<mkldnn::reduction::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::reduction> fwd_;

private:
  void Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
            const mkldnn::algorithm alg_kind, const bool with_workspace) {
    const auto src_md = input.GetMKLDNNData()->get_desc();
    const auto dst_md = GetMemDesc(output);
    const mkldnn::engine engine = CpuEngine::Get()->get_engine();
///dd    std::cout << "EEEEEE\n";
    const auto fwd_desc = mkldnn::reduction::desc(
        mkldnn::algorithm::reduction_norm_lp_power_p_sum, src_md, dst_md,
        /*power:=*/2.f, (float)(param_.eps));
//    std::cout << "GGGGGGGGGGGGG\n";
    this->fwd_pd_.reset(
        new mkldnn::reduction::primitive_desc(fwd_desc, engine));
 //   std::cout << "vvvv\n";
    this->fwd_.reset(new mkldnn::reduction(*(this->fwd_pd_)));
   // std::cout << "GasdsadsadsadsadG\n";
  }

  void Execute(const mxnet::NDArray &input, const mxnet::NDArray &output,
               const OpReqType req) {
    //std::cout << "DDDDDDD22222\n";
    NDArray in_buffer = input;
    if (input.IsView() && input.IsMKLDNNData()) {
      in_buffer = input.Reorder2Default();
    }
   // std::cout << "DDDDDDD21312321\n";
    auto input_mem = in_buffer.GetMKLDNNData();
  //  std::cout << "DDDDDDD333333333333\n";
    auto output_mem_t = CreateMKLDNNMem(output, this->fwd_pd_->dst_desc(), req, &input);
    // mkldnn_args_map_t args = {
    //{MKLDNN_ARG_SRC, *input_mem},
    //{MKLDNN_ARG_DST, *(output_mem_t.second) }
    // };
    std::cout << "DDDDDDD11111\n";
    //  MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
    // CommitOutput(output, output_mem_t);
    std::cout << "DDDDDDD\n";
    // MKLDNNStream::Get()->Submit();
    // std::cout << "AAAAAAA\n";
  }

  MKLDNNL2_NormalizationOpCPU(const mxnet::NDArray &input,
                              const mxnet::NDArray &output,
                              const mkldnn::algorithm alg_kind,
                              const bool with_workspace)
      : with_workspace_(with_workspace), fwd_pd_(nullptr) {
    Init(input, output, alg_kind, with_workspace);
  }

  MKLDNNL2_NormalizationOpCPU &GetL2Fwd(const mxnet::NDArray &input,
                                        const mxnet::NDArray &output) {
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
    /*bool with_workspace = true;
    MKLDNNL2Signature key(param_);
    key.AddSign(with_workspace);
    key.AddSign(input);
    key.AddSign(output);
    auto it = l2_fwds.find(key);
    if (it == l2_fwds.end()) {*/
    auto data_md = input.GetMKLDNNData()->get_desc();
    mkldnn::algorithm kind = mkldnn::algorithm::reduction_norm_lp_power_p_sum;
    MKLDNNL2_NormalizationOpCPU fwd(input, output, kind, false);
    // it = AddToCache(&l2_fwds, key, fwd);
    // }
    return fwd;
  }

  void Compute(const mxnet::NDArray &input, const mxnet::NDArray &output,
               const OpReqType req) {
    auto &fwd = GetL2Fwd(input, output);
    fwd.Execute(input, output, req);
  }
};

} // namespace op
} // namespace mxnet
#endif // MXNET_OPERATOR_MKLDNN_L2_NORMALIZATION_INL_H_