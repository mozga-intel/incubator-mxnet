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
 * \file mkldnn_pooling-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <mkldnn.hpp>
#include "../../operator_common.h"
#include "./mkldnn_base-inl.h"

template<typename TH> void _dbg(const char* sdbg, TH h) { std::cerr<<sdbg<<"="<<h<<std::endl; }
template<typename TH, typename... TA> void _dbg(const char* sdbg, TH h, TA... t) {
      while(*sdbg != ',') { std::cerr<<*sdbg++; } std::cerr<<"="<<h<<","; _dbg(sdbg+1, t...);
}
#ifdef LOCAL
#define debug(...) _dbg(#__VA_ARGS__, __VA_ARGS__)
#define debugv(x) {{std::cerr <<#x <<" = "; FORE(itt, (x)) std::cerr <<*itt <<", "; std::cerr <<endl; }}
//#else
//#define debug(...) (__VA_ARGS__)
//#define debugv(x)
//#define cerr if(0)std::cout
#endif

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
                const bool with_workspace, const bool is_train):
            with_workspace_(with_workspace), fwd_(nullptr) {
                Init(input, output, kernel, strides, pad_l, pad_r, is_train, alg_kind);
            }
        ~MKLDNNAdaptivePoolingFwd() { }
        void Execute() { }
        void Execute(const NDArray &input,
                const OpReqType req,
                const NDArray &output,
                const NDArray *workspace) {
            NDArray in_buffer = input;
            if(input.IsView() && input.IsMKLDNNData()) in_buffer = input.Reorder2Default();
            auto input_mem = in_buffer.GetMKLDNNData();
            auto output_mem_t = CreateMKLDNNMem(output, this->fwd_pd_->dst_desc(), req);
            mkldnn_args_map_t args = {
                {MKLDNN_ARG_SRC, *input_mem},
                {MKLDNN_ARG_DST, *(output_mem_t.second) }
            };
            if(this->with_workspace_) {
                auto engine = CpuEngine::Get()->get_engine();
                if(workspace == nullptr) LOG(FATAL) << "MKLDNN Average Pooling: incorrect worskapce input";
                auto ws = std::make_shared<mkldnn::memory>((*(this->fwd_pd_)).workspace_desc(),
                        engine, workspace->GetMKLDNNData()->get_data_handle());
                args[MKLDNN_ARG_WORKSPACE] = *ws;
            }
            if(this->fwd_) {
                MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
                CommitOutput(output, output_mem_t);
                MKLDNNStream::Get()->Submit();
            } else { 
                LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
            }
        }
    private:
        bool with_workspace_;
        std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_; 
        std::shared_ptr<mkldnn::pooling_forward> fwd_;
    private:
        void Init(const mxnet::NDArray &input,
                const mxnet::NDArray &output,
                const mkldnn::memory::dims &kernel,
                const mkldnn::memory::dims &strides,
                const mkldnn::memory::dims &pad_l,
                const mkldnn::memory::dims &pad_r,
                const bool is_train, const mkldnn::algorithm alg_kind) {
            const auto src_md = input.GetMKLDNNData()->get_desc();
            const auto dst_md = GetMemDesc(output);
            const mkldnn::engine engine = CpuEngine::Get()->get_engine(); 
            if(alg_kind != mkldnn::algorithm::pooling_avg &&
               alg_kind != mkldnn::algorithm::pooling_avg_include_padding &&
               alg_kind != mkldnn::algorithm::pooling_avg_exclude_padding) {
                LOG(FATAL) << "MKLDNN Adaptive Pooling: algorithm is not supported";
            }
            mkldnn::prop_kind prop = mkldnn::prop_kind::forward_scoring;
            if(is_train && alg_kind != mkldnn::algorithm::pooling_avg) {
                prop = mkldnn::prop_kind::forward_training;
            }
            if (is_train && prop == mkldnn::prop_kind::forward_scoring) {
                LOG(INFO) << "MKLDNN Pooling: training with prop_kind is forward_scoring";
            }
            const auto fwd_desc = mkldnn::pooling_forward::desc(prop, alg_kind, src_md, dst_md, strides, kernel, pad_l, pad_r);
            this->fwd_pd_.reset(new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
            this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_)));
        }
};

template<typename T>
MKLDNNAdaptivePoolingFwd &GetPoolingFwd(const T &param,
        const bool is_train,
        const NDArray &input,
        const NDArray &output) {
    if(input.shape().ndim() != 4) {
        LOG(FATAL) << "MKLDNN Adaptive Avg Pool 2d: Expect only 2D input";
    }
    typedef ParamOpSign<T> MKLDNNPoolingSignature;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNAdaptivePoolingFwd,
                                         OpHash> pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNAdaptivePoolingFwd,
                                            OpHash> pooling_fwds;
#endif
   bool with_workspace = is_train && true;
   MKLDNNPoolingSignature key(param);
   key.AddSign(is_train);
   key.AddSign(with_workspace);
   key.AddSign(input);
   key.AddSign(output);

   auto it = pooling_fwds.find(key);
   if(it == pooling_fwds.end()) {
       auto data_md = input.GetMKLDNNData()->get_desc();
       const int kernel_ndims = input.shape().ndim();

       mkldnn::memory::dims kernel(kernel_ndims);
       mkldnn::memory::dims strides(kernel_ndims);
       mkldnn::memory::dims pad_l(kernel_ndims);
       mkldnn::memory::dims pad_r(kernel_ndims);

       auto update_kernel = [&](mkldnn::memory::dims &kernel, const NDArray &in_data, const NDArray &out_data) {
           for(int64_t idx = 2; idx < in_data.shape().ndim(); ++idx) {
               const auto s1 = in_data.shape()[idx];
               auto s2 = out_data.shape()[idx];
               if(s2 == 0) { LOG(FATAL) << "Output size can not be zero"; }
               if(s1 % s2 != 0) {
                   LOG(FATAL) << "Input size is not divisible by the output size  s1 mod s2 != 0"; 
               }
               kernel[idx-2] = s1 / s2;
           }
       };

       auto update_kernel_v2 = [&](mkldnn::memory::dims &kernel,
               mkldnn::memory::dims &strides,
               mkldnn::memory::dims &pad_l,
               mkldnn::memory::dims &pad_r,
               const NDArray &in_data, const NDArray &out_data) {
           const int IH = in_data.shape()[2];
           const int IW = in_data.shape()[3];
           const int OH = out_data.shape()[2];
           const int OW = out_data.shape()[3];

           strides.at(0) = floor((IH << 1) / OH) - floor(IH / OH);
           strides.at(1) = floor((IW << 1) / OW) - floor(IW / OW);
           kernel.at(0) = ceil((IH << 1) / OH) - floor(IH / OH);
           kernel.at(1) = ceil((IW << 1) / OW) - floor(IW / OW);
           pad_l.at(0) = (strides.at(0) * (OH - 1) + kernel.at(0) - IH) / 2;
           pad_l.at(1) = (strides.at(1) * (OW - 1) + kernel.at(1) - IW) / 2;
       };

       auto update_padding = [&](mkldnn::memory::dims &kernel, int input_dim) {
           for(int64_t idx = 0; idx < input_dim - 2; ++idx) {
               kernel[idx] = 0;
           }
       };

       auto update_strides = [&](mkldnn::memory::dims &kernel, const NDArray &in_data) {
           auto get_stride = [&](const NDArray &tensor, int idx) {
               int stride = 1;
               const int Dim = tensor.shape().ndim();
               for (int i = Dim-2; i >= idx; --i) {
                   stride *= tensor.shape()[i+1];
               }
               return stride;
           };
           for(int64_t idx = 0; idx  < in_data.shape().ndim() - 2; ++idx) {
               kernel[idx] = get_stride(in_data, idx);
           }
       };

       //update_kernel(kernel, input, output);
       //update_padding(pad_l, input.shape().ndim());
       //update_padding(pad_r, input.shape().ndim());
       //update_strides(strides, input);
       update_kernel_v2(kernel, strides, pad_l, pad_r, input, output);
       mkldnn::memory::validate_dims(kernel);
       mkldnn::memory::validate_dims(strides);
       mkldnn::memory::validate_dims(pad_l);
       mkldnn::memory::validate_dims(pad_r);

       mkldnn::algorithm kind = mkldnn::algorithm::pooling_avg;
       MKLDNNAdaptivePoolingFwd fwd(input, output, kernel, kernel, pad_l, pad_r, kind, false, false);
       it = AddToCache(&pooling_fwds, key, fwd);
   }
   return it->second;
}

template<typename T>
void MKLDNNAdaptivePoolingCompute(const OpContext &ctx, const T &param,
                          const NDArray &in_data, const OpReqType req,
                          const NDArray &out_data, const NDArray *workspace) {
    auto &fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
    fwd.Execute(in_data, req, out_data, workspace);
}

template<typename T>
void MKLDNNAdaptivePoolingGradCompute(const OpContext &ctx, const T &param,
                              const NDArray &out_grad, const NDArray &in_data,
                              const NDArray *workspace, const OpReqType req,
                              const NDArray &in_grad);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_
