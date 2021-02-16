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
 * \file error_utils-inl.h
 * \brief
 * \author Mateusz Ozga
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_ADAPTIVE_POOLING_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_ADAPTIVE_POOLING_INL_H_

#include <type_traits>
#include <utility>

//#if MXNET_USE_MKLDNN == 1
namespace details {
inline namespace v1 {

template<typename...> 
using void_t = void;

template<typename TF>
struct validity_checker {
  template<typename... Ts>
  constexpr auto operator()(Ts&&...) const {
    return is_callable<TF(Ts...)>{};
  }
};

template<typename, typename = void>
struct is_callable : std::false_type 
{ 
};

template <typename TF, typename... TArgs>
struct is_callable<TF(TArgs...),
    void_t<decltype(std::declval<TF>()(std::declval<TArgs>()...))>>
    : std::true_type
{
};

template<typename TF>
constexpr auto is_valid(TF) {
  return validity_checker<TF>{};
}


} // inline namespace v1;
} // namespace details
//#endif // MXNET_USE_MKLDNN == 1
#endif //MXNET_OPERATOR_NN_MKLDNN_ADAPTIVE_POOLING_INL_H_