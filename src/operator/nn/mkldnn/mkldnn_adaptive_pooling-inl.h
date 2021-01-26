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
//#include "../../contrib/adaptive_avg_pooling-inl.h"
#include "./mkldnn_base-inl.h"

template<typename TH> void _dbg(const char* sdbg, TH h) { cerr<<sdbg<<"="<<h<<endl; }
template<typename TH, typename... TA> void _dbg(const char* sdbg, TH h, TA... t) {
      while(*sdbg != ',') { cerr<<*sdbg++; } cerr<<"="<<h<<","; _dbg(sdbg+1, t...);
}
#ifdef LOCAL
#define debug(...) _dbg(#__VA_ARGS__, __VA_ARGS__)
#define debugv(x) {{cerr <<#x <<" = "; FORE(itt, (x)) cerr <<*itt <<", "; cerr <<endl; }}
#else
#define debug(...) (__VA_ARGS__)
#define debugv(x)
#define cerr if(0)cout
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
        void Execute(const NDArray &input,
                const OpReqType req,
                const NDArray &output,
                const NDArray *workspace) {
            NDArray in_buffer = input;
            if(input.IsView() && input.IsMKLDNNData())
                in_buffer = input.Reorder2Default();
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
        }
};

inline int get_stride(const NDArray &tensor, int idx) {
    int stride = 1;
    int Dim = tensor.shape().ndim();
    for (int i = Dim-2; i >= idx; --i) {
        stride *= tensor.shape()[i+1];
    }
    return stride;
}

template<typename T>
MKLDNNAdaptivePoolingFwd &GetPoolingFwd(const T &param,
        const bool is_train,
        const NDArray &input,
        const NDArray &output) {
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
   // TODO is always true
   bool with_workspace = is_train && true;
   MKLDNNPoolingSignature key(param);
   key.AddSign(is_train);
   key.AddSign(with_workspace);
   key.AddSign(input);
   key.AddSign(output);

   auto it = pooling_fwds.find(key);
   if(it == pooling_fwds.end()) {
       // This kernel is not implemented;
       auto data_md = input.GetMKLDNNData()->get_desc();
       const int kernel_ndims = input.shape().ndim();

       // MKDLNN dims have been created;
       mkldnn::memory::dims kernel(kernel_ndims);
       mkldnn::memory::dims strides(kernel_ndims);
       mkldnn::memory::dims pad_l(kernel_ndims);
       mkldnn::memory::dims pad_r(kernel_ndims);

       int64_t sizeB = input.shape()[0];
       int64_t sizeD  = input.shape()[1];
       int64_t isizeH = input.shape()[2];
       int64_t isizeW = input.shape()[3];

       int64_t istrideB = get_stride(input, 0);
       int64_t istrideD = get_stride(input, 1);
       int64_t istrideH = get_stride(input, 2);
       int64_t istrideW = get_stride(input, 3);

       int64_t osizeH = output.shape()[2];
       int64_t osizeW = output.shape()[3];

   
       std::cout << "kernel_out_size = " << output.shape().ndim() << std::endl; 
       std::cout << "Kernel_in_size = " <<  input.shape().ndim() << std::endl;
       std::cout << "sizeB: " << sizeB << " sizeD: " << sizeD << " iSzieH: " << isizeH << " isize: = " << isizeW << std::endl; 
       std::cout << "istrideB: " << istrideB << " istrideD:" << istrideD << " istrideH" << istrideH << " istrideW" << istrideW << std::endl;
       std::cout << "osize:" << osizeH << " osizeW:" << osizeW << std::endl;

       if(kernel_ndims == 1) {  }
       if(kernel_ndims == 1) {  }
       if(kernel_ndims == 2) {  }
       if(kernel_ndims == 3) {  }
       if(kernel_ndims == 4) {
           kernel[0] = sizeB;
           kernel[1] = sizeD;
           kernel[2] = isizeH;
           kernel[3] = isizeW;

           strides[0] = istrideB;
           strides[1] = istrideD;
           strides[2] = istrideH;
           strides[3] = istrideW;

           pad_l[0] = 0;
           pad_l[1] = 0;
           pad_l[2] = 0;
           pad_l[3] = 0;

           pad_r[0] = 0;
           pad_r[1] = 0;
           pad_r[2] = 0;
           pad_r[3] = 0;
       }
       mkldnn::algorithm kind = mkldnn::algorithm::pooling_avg;
       MKLDNNAdaptivePoolingFwd fwd(input, output, kernel, strides, pad_l, pad_r, kind, false, false); 
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
