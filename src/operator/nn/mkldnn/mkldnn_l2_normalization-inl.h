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

#include <mxnet/operator.h>
#include <mkldnn.hpp>

namespace mxnet {
namespace op {

template<typename T> 
struct Is {
    const T d_;
    template<class... Args> bool in(Args ...args) { 
        bool r {false};
        [](...){}(( ( r = r || d_ == args), 1)...);
        return r;
    }
};

template<class T>
Is<T> is(T d) {
    return Is<T>{d};
}

template<typename DType, typename TParam>
class MKLDNNL2_NormalizationOpCPU : public Operator {
    public:
        explicit MKLDNNL2_NormalizationOpCPU(TParam p) { 
            this->param_ = p;
        }

        virtual void Forward(const OpContext &ctx,
                             const std::vector<TBlob> &in_data,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &out_data,
                             const std::vector<TBlob> &aux_args) {
            CHECK_EQ(in_data.size(), 1U);
            CHECK_EQ(out_data.size(), 2U);

            constexpr int device_index = 1;
            NDArray input_data(in_data[0], device_index);
            NDArray output_data(out_data[0], device_index);

            const auto src_md = input_data.GetMKLDNNData()->get_desc();
            const auto dst_md = GetMemDesc(output_data);
            const mkldnn::engine engine = CpuEngine::Get()->get_engine();

            const auto fwd_desc = mkldnn::reduction::desc(
                mkldnn::algorithm::reduction_norm_lp_power_p_sum,
                src_md, dst_md, /*power:=*/ 2, param_.eps);

            this->fwd_pd_.reset(new mkldnn::reduction::primitive_desc(fwd_desc, engine));
            this->fwd_.reset(new mkldnn::reduction(*(this->fwd_pd_)));

            auto output_mem_t = CreateMKLDNNMem(output_data, this->fwd_pd_->dst_desc(), req[0]);
            auto input_mem = input_data.GetMKLDNNData();

            mkldnn_args_map_t args = {
                {MKLDNN_ARG_SRC, *input_mem},
                {MKLDNN_ARG_DST, *(output_mem_t.second) }
            };

            MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
            CommitOutput(output_data, output_mem_t);
            MKLDNNStream::Get()->Submit();
        }

    protected:
        TParam param_;
        std::shared_ptr<mkldnn::reduction::primitive_desc> fwd_pd_; 
        std::shared_ptr<mkldnn::reduction> fwd_;
};

}  // namespace op
}  // namespace mxnet
#endif // MXNET_OPERATOR_MKLDNN_L2_NORMALIZATION_INL_H_
