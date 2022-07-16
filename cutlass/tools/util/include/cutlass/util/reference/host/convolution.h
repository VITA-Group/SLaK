/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Reference implementation for convolution in host-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "./gemm.h"

namespace cutlass {
namespace reference {
namespace host {

namespace detail {
template <typename T, typename S>
struct need_round {
    static bool const src_float =
            cutlass::platform::is_floating_point<S>::value;
    static bool const dst_integer =
            cutlass::platform::is_integral<T>::value ||
            cutlass::platform::is_same<T, int8_t>::value ||
            cutlass::platform::is_same<T, uint8_t>::value ||
            cutlass::platform::is_same<T, cutlass::int4b_t>::value ||
            cutlass::platform::is_same<T, cutlass::uint4b_t>::value;
    static bool const value = src_float && dst_integer;
};

template <typename T>
struct need_clamp {
    static bool const value =
            cutlass::platform::is_integral<T>::value ||
            cutlass::platform::is_same<T, int8_t>::value ||
            cutlass::platform::is_same<T, uint8_t>::value ||
            cutlass::platform::is_same<T, cutlass::int4b_t>::value ||
            cutlass::platform::is_same<T, cutlass::uint4b_t>::value;
};

template <typename T>
static inline T round(T val) {
    return std::round(val);
}

template <>
inline cutlass::half_t round(cutlass::half_t val) {
    return cutlass::half_t(std::round(static_cast<float>(val)));
}
}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Forward propagation
////////////////////////////////////////////////////////////////////////////////////////////////////

/// y = conv2d(x, w)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv2dFprop(conv::Conv2dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_x,
                 TensorRef<ElementB, LayoutB> tensor_w,
                 TensorRef<ElementC, LayoutC> tensor_y_in,
                 TensorRef<ElementC, LayoutC> tensor_y_out,
                 ElementCompute alpha, ElementCompute beta) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < problem_size.N; ++n) {
        for (int p = 0; p < problem_size.P; ++p) {
            for (int q = 0; q < problem_size.Q; ++q) {
                for (int k = 0; k < problem_size.K; ++k) {
                    ElementAccumulator acc = ElementAccumulator();

                    for (int r = 0; r < problem_size.R; ++r) {
                        for (int s = 0; s < problem_size.S; ++s) {
                            for (int c = 0; c < problem_size.C; ++c) {
                                int filter_r = r;
                                int filter_s = s;

                                if (problem_size.mode ==
                                    cutlass::conv::Mode::kConvolution) {
                                    filter_r = problem_size.R - 1 - r;
                                    filter_s = problem_size.S - 1 - s;
                                }

                                int h = p * problem_size.stride_h -
                                        problem_size.pad_h +
                                        filter_r * problem_size.dilation_h;
                                int w = q * problem_size.stride_w -
                                        problem_size.pad_w +
                                        filter_s * problem_size.dilation_w;

                                if (h >= 0 && h < problem_size.H && w >= 0 &&
                                    w < problem_size.W) {
                                    ElementA a = tensor_x.at({n, h, w, c});
                                    ElementB b = tensor_w.at({k, r, s, c});

                                    acc = inner_product_op(
                                            ElementAccumulator(a),
                                            ElementAccumulator(b), acc);
                                }
                            }
                        }
                    }

                    // Apply Epilogue, compute ElementCompute, convert and store
                    // ElementC
                    ElementC c_ref = ElementC();

                    if (beta != ElementCompute()) {
                        c_ref = tensor_y_in.at(cutlass::make_Coord(n, p, q, k));
                    }

                    tensor_y_out.at(cutlass::make_Coord(n, p, q, k)) =
                            convert_op(alpha * ElementCompute(acc) +
                                       beta * ElementCompute(c_ref));
                }
            }
        }
    }
}

/// Depthwise-separable convolution
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementAccumulator, typename ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Depsep_Fprop(
        cutlass::TensorView<ElementA, LayoutA> tensor_A,
        cutlass::TensorView<ElementB, LayoutB> tensor_B,
        cutlass::TensorView<ElementC, LayoutC> tensor_C, ElementCompute alpha,
        ElementCompute beta, cutlass::Tensor4DCoord padding,
        cutlass::Coord<2> conv_stride, cutlass::Coord<2> dilation,
        cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < tensor_C.extent().n(); ++n) {
        for (int p = 0; p < tensor_C.extent().h(); ++p) {
            for (int q = 0; q < tensor_C.extent().w(); ++q) {
                for (int g = 0; g < tensor_C.extent().c(); ++g) {
                    ElementAccumulator acc = ElementAccumulator();
                    for (int r = 0; r < tensor_B.extent().h(); ++r) {
                        for (int s = 0; s < tensor_B.extent().w(); ++s) {
                            if ((p * conv_stride[0] - padding[0] +
                                 r * dilation[0]) < tensor_A.extent().h() &&
                                (p * conv_stride[0] - padding[0] +
                                 r * dilation[0]) >= 0 &&
                                (q * conv_stride[1] - padding[2] +
                                 s * dilation[1]) < tensor_A.extent().w() &&
                                (q * conv_stride[1] - padding[2] +
                                 s * dilation[1]) >= 0) {
                                ElementA a = tensor_A.at(cutlass::make_Coord(
                                        n,
                                        p * conv_stride[0] - padding[0] +
                                                r * dilation[0],
                                        q * conv_stride[1] - padding[2] +
                                                s * dilation[1],
                                        g));

                                ElementB b =
                                        (mode ==
                                         cutlass::conv::Mode::kCrossCorrelation)
                                                ? tensor_B.at(
                                                          cutlass::make_Coord(
                                                                  g, r, s, 0))
                                                : tensor_B.at(cutlass::make_Coord(
                                                          g,
                                                          tensor_B.extent()
                                                                          .h() -
                                                                  r - 1,
                                                          tensor_B.extent()
                                                                          .w() -
                                                                  s - 1,
                                                          0));

                                acc = inner_product_op(ElementAccumulator(a),
                                                       ElementAccumulator(b),
                                                       acc);
                            }
                        }
                    }

                    // Apply Epilogue, compute ElementCompute, convert and store
                    // ElementC
                    ElementC c_ref =
                            tensor_C.at(cutlass::make_Coord(n, p, q, g));
                    tensor_C.at(cutlass::make_Coord(n, p, q, g)) =
                            convert_op(alpha * ElementCompute(acc) +
                                       beta * ElementCompute(c_ref));
                }
            }
        }
    }
}

/// Depthwise-separable convolution
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias,
          typename ElementAccumulator, typename ElementCompute,
          typename ConvertOp = NumericConverter<ElementDst, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Depsep_Fprop(
        conv::Conv2dProblemSize conv_param,
        cutlass::TensorRef<ElementSrc, LayoutSrc> tensor_src,
        cutlass::TensorRef<ElementFilter, LayoutFilter> tensor_filter,
        cutlass::TensorRef<ElementBias, LayoutBias> tensor_bias,
        cutlass::TensorRef<ElementDst, LayoutDst> tensor_z,
        cutlass::TensorRef<ElementDst, LayoutDst> tensor_dst,
        ElementCompute alpha, ElementCompute beta, ElementCompute gamma,
        cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    int const N = conv_param.N;
    int const G = conv_param.K;
    int const H = conv_param.H;
    int const W = conv_param.W;
    int const P = conv_param.P;
    int const Q = conv_param.Q;
    int const R = conv_param.R;
    int const S = conv_param.S;
    int const PH = conv_param.pad_h;
    int const PW = conv_param.pad_w;
    int const SH = conv_param.stride_h;
    int const SW = conv_param.stride_w;
    int const DH = conv_param.dilation_h;
    int const DW = conv_param.dilation_w;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < P; ++p) {
            for (int q = 0; q < Q; ++q) {
                for (int g = 0; g < G; ++g) {
                    ElementAccumulator acc = ElementAccumulator();
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            if ((p * SH - PH + r * DH) < H &&
                                (p * SH - PH + r * DH) >= 0 &&
                                (q * SW - PW + s * DW) < W &&
                                (q * SW - PW + s * DW) >= 0) {
                                ElementSrc sv = tensor_src.at(
                                        {n, p * SH - PH + r * DH,
                                         q * SW - PW + s * DW, g});

                                ElementFilter fv =
                                        (mode ==
                                         cutlass::conv::Mode::kCrossCorrelation)
                                                ? tensor_filter.at({g, r, s, 0})
                                                : tensor_filter.at(
                                                          {g, R - r - 1,
                                                           S - s - 1, 0});

                                acc = inner_product_op(ElementAccumulator(sv),
                                                       ElementAccumulator(fv),
                                                       acc);
                            }
                        }
                    }

                    // Apply Epilogue, compute ElementCompute, convert and store
                    // ElementC
                    ElementCompute intermediate = alpha * ElementCompute(acc);
                    if (beta != ElementCompute()) {
                        intermediate += beta * tensor_bias.at({0, 0, 0, g});
                    }
                    if (gamma != ElementCompute()) {
                        intermediate += gamma * tensor_z.at({n, p, q, g});
                    }
                    if (detail::need_round<ElementDst, ElementCompute>::value) {
                        intermediate = detail::round(intermediate);
                    }
                    tensor_dst.at({n, p, q, g}) = convert_op(intermediate);
                }
            }
        }
    }
}

/// Depthwise-separable convolution
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias,
          typename ElementAccumulator, typename ElementCompute,
          typename ConvertOp = NumericConverter<ElementDst, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Depsep_Dgrad(
        conv::Conv2dProblemSize conv_param,
        cutlass::TensorRef<ElementSrc, LayoutSrc> tensor_src,
        cutlass::TensorRef<ElementFilter, LayoutFilter> tensor_filter,
        cutlass::TensorRef<ElementBias, LayoutBias> tensor_bias,
        cutlass::TensorRef<ElementDst, LayoutDst> tensor_z,
        cutlass::TensorRef<ElementDst, LayoutDst> tensor_dst,
        ElementCompute alpha, ElementCompute beta, ElementCompute gamma,
        cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    int const N = conv_param.N;
    int const G = conv_param.K;
    int const H = conv_param.H;
    int const W = conv_param.W;
    int const P = conv_param.P;
    int const Q = conv_param.Q;
    int const R = conv_param.R;
    int const S = conv_param.S;
    int const PH = conv_param.pad_h;
    int const PW = conv_param.pad_w;
    int const SH = conv_param.stride_h;
    int const SW = conv_param.stride_w;
    int const DH = conv_param.dilation_h;
    int const DW = conv_param.dilation_w;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int g = 0; g < G; ++g) {
                    ElementAccumulator acc = ElementAccumulator();
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int filter_r = r;
                            int filter_s = s;
                            if (conv_param.mode ==
                                cutlass::conv::Mode::kConvolution) {
                                filter_r = R - 1 - r;
                                filter_s = S - 1 - s;
                            }

                            int p = h + PH - filter_r * DH;
                            int q = w + PW - filter_s * DW;

                            if (p >= 0 && (p % SH) == 0 && q >= 0 &&
                                (q % SW) == 0) {
                                p = p / SH;
                                q = q / SW;

                                if (p < P && q < Q) {
                                    ElementSrc sv = tensor_src.at({n, p, q, g});

                                    ElementFilter fv = tensor_filter.at(
                                            {g, filter_r, filter_s, 0});
                                    acc = inner_product_op(
                                            ElementAccumulator(sv),
                                            ElementAccumulator(fv), acc);
                                }
                            }
                        }
                    }

                    // Apply Epilogue, compute ElementCompute, convert and
                    // store ElementC
                    ElementCompute intermediate = alpha * ElementCompute(acc);
                    if (beta != ElementCompute()) {
                        intermediate +=
                                beta *
                                ElementCompute(tensor_bias.at({0, 0, 0, g}));
                    }
                    if (gamma != ElementCompute()) {
                        intermediate +=
                                gamma *
                                ElementCompute(tensor_z.at({n, h, w, g}));
                    }
                    if (detail::need_round<ElementDst, ElementCompute>::value) {
                        intermediate = detail::round(intermediate);
                    }
                    tensor_dst.at({n, h, w, g}) = convert_op(intermediate);
                }
            }
        }
    }
}

/// Depthwise-separable convolution
template <typename ElementSrc, typename LayoutSrc, typename ElementDiff,
          typename LayoutDiff, typename ElementGrad, typename LayoutGrad,
          typename ElementAccumulator, typename ElementCompute,
          typename ConvertOp = NumericConverter<ElementGrad, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Depsep_Wgrad(
        conv::Conv2dProblemSize conv_param,
        cutlass::TensorRef<ElementSrc, LayoutSrc> tensor_src,
        cutlass::TensorRef<ElementDiff, LayoutDiff> tensor_diff,
        cutlass::TensorRef<ElementGrad, LayoutGrad> tensor_grad,
        ElementCompute alpha,
        cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    int const N = conv_param.N;
    int const G = conv_param.K;
    int const H = conv_param.H;
    int const W = conv_param.W;
    int const P = conv_param.P;
    int const Q = conv_param.Q;
    int const R = conv_param.R;
    int const S = conv_param.S;
    int const PH = conv_param.pad_h;
    int const PW = conv_param.pad_w;
    int const SH = conv_param.stride_h;
    int const SW = conv_param.stride_w;
    int const DH = conv_param.dilation_h;
    int const DW = conv_param.dilation_w;

    // Apply MMA and accumulate ElementAccumulator
    for (int g = 0; g < G; ++g) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                ElementAccumulator acc = ElementAccumulator();
                for (int n = 0; n < N; ++n) {
                    for (int p = 0; p < P; ++p) {
                        for (int q = 0; q < Q; ++q) {
                            int filter_r = r;
                            int filter_s = s;
                            if (conv_param.mode ==
                                cutlass::conv::Mode::kConvolution) {
                                filter_r = R - 1 - r;
                                filter_s = S - 1 - s;
                            }

                            int h = p * SH - PH + filter_r * DH;
                            int w = q * SW - PW + filter_s * DW;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                ElementSrc sv = tensor_src.at({n, h, w, g});

                                ElementDiff dv = tensor_diff.at({n, p, q, g});
                                acc = inner_product_op(ElementAccumulator(sv),
                                                       ElementAccumulator(dv),
                                                       acc);
                            }
                        }
                    }
                }

                // Apply Epilogue, compute ElementCompute, convert and
                // store ElementC
                ElementCompute intermediate = alpha * ElementCompute(acc);
                if (detail::need_round<ElementGrad, ElementCompute>::value) {
                    intermediate = detail::round(intermediate);
                }
                tensor_grad.at({g, r, s, 0}) = convert_op(intermediate);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dgrad
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dx = dgrad(dy, w)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv2dDgrad(cutlass::conv::Conv2dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_dy,
                 TensorRef<ElementB, LayoutB> tensor_w,
                 TensorRef<ElementC, LayoutC> tensor_dx_in,
                 TensorRef<ElementC, LayoutC> tensor_dx_out,
                 ElementCompute alpha, ElementCompute beta) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < problem_size.N; ++n) {
        for (int h = 0; h < problem_size.H; ++h) {
            for (int w = 0; w < problem_size.W; ++w) {
                for (int c = 0; c < problem_size.C; ++c) {
                    ElementAccumulator acc = ElementAccumulator();

                    for (int r = 0; r < problem_size.R; ++r) {
                        for (int s = 0; s < problem_size.S; ++s) {
                            for (int k = 0; k < problem_size.K; ++k) {
                                int filter_r = r;
                                int filter_s = s;

                                if (problem_size.mode ==
                                    cutlass::conv::Mode::kConvolution) {
                                    filter_r = problem_size.R - 1 - r;
                                    filter_s = problem_size.S - 1 - s;
                                }

                                int p = h + problem_size.pad_h -
                                        filter_r * problem_size.dilation_h;
                                int q = w + problem_size.pad_w -
                                        filter_s * problem_size.dilation_w;

                                if (p >= 0 &&
                                    (p % problem_size.stride_h) == 0 &&
                                    q >= 0 &&
                                    (q % problem_size.stride_w) == 0) {
                                    p = p / problem_size.stride_h;
                                    q = q / problem_size.stride_w;

                                    if (p < problem_size.P &&
                                        q < problem_size.Q) {
                                        ElementA a = tensor_dy.at(
                                                cutlass::make_Coord(n, p, q,
                                                                    k));
                                        ElementB b =
                                                tensor_w.at(cutlass::make_Coord(
                                                        k, r, s, c));

                                        acc = inner_product_op(
                                                ElementAccumulator(a),
                                                ElementAccumulator(b), acc);
                                    }
                                }

                            }  // for (K)
                        }      // for (S)
                    }          // for (R)

                    // Apply Epilogue, compute ElementCompute, convert
                    // and store ElementC
                    ElementC c_ref = ElementC();

                    if (beta != ElementCompute()) {
                        c_ref = tensor_dx_in.at(
                                cutlass::make_Coord(n, h, w, c));
                    }

                    tensor_dx_out.at(cutlass::make_Coord(n, h, w, c)) =
                            convert_op(alpha * ElementCompute(acc) +
                                       beta * ElementCompute(c_ref));

                }  // for (C)
            }      // for (W)
        }          // for (H)
    }              // for (N)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wgrad
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dw = wgrad(dy, x)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv2dWgrad(cutlass::conv::Conv2dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_dy,
                 TensorRef<ElementB, LayoutB> tensor_x,
                 TensorRef<ElementC, LayoutC> tensor_dw_in,
                 TensorRef<ElementC, LayoutC> tensor_dw_out,
                 ElementCompute alpha, ElementCompute beta) {
    InnerProductOp inner_product_op;
    ConvertOp convert_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int k = 0; k < problem_size.K; ++k) {
        for (int r = 0; r < problem_size.R; ++r) {
            for (int s = 0; s < problem_size.S; ++s) {
                for (int c = 0; c < problem_size.C; ++c) {
                    ElementAccumulator acc = ElementAccumulator();

                    for (int n = 0; n < problem_size.N; ++n) {
                        for (int p = 0; p < problem_size.P; ++p) {
                            for (int q = 0; q < problem_size.Q; ++q) {
                                cutlass::Tensor4DCoord b_coord;

                                int filter_r = r;
                                int filter_s = s;

                                if (problem_size.mode ==
                                    cutlass::conv::Mode::kConvolution) {
                                    filter_r = problem_size.R - 1 - r;
                                    filter_s = problem_size.S - 1 - s;
                                }

                                b_coord = make_Coord(
                                        n,
                                        p * problem_size.stride_h -
                                                problem_size.pad_h +
                                                filter_r *
                                                        problem_size.dilation_h,
                                        q * problem_size.stride_w -
                                                problem_size.pad_w +
                                                filter_s *
                                                        problem_size.dilation_w,
                                        c);

                                if (b_coord.h() < problem_size.H &&
                                    b_coord.h() >= 0 &&
                                    b_coord.w() < problem_size.W &&
                                    b_coord.w() >= 0) {
                                    ElementAccumulator a = ElementAccumulator(
                                            tensor_dy.at(cutlass::make_Coord(
                                                    n, p, q, k)));
                                    ElementAccumulator b = ElementAccumulator(
                                            tensor_x.at(b_coord));
                                    acc = inner_product_op(a, b, acc);
                                }
                            }
                        }
                    }

                    // Apply Epilogue, compute ElementCompute, convert
                    // and store ElementC
                    ElementC c_ref = ElementC();

                    if (beta != ElementCompute()) {
                        c_ref = tensor_dw_in.at(
                                cutlass::make_Coord(k, r, s, c));
                    }

                    tensor_dw_out.at(cutlass::make_Coord(k, r, s, c)) =
                            convert_op(alpha * ElementCompute(acc) +
                                       beta * ElementCompute(c_ref));

                }  // for (C)
            }      // for (S)
        }          // for (R)
    }              // for (K)
}

/// Generic 2D convolution targeting Conv2dFprop, Conv2dDgrad, and
/// Conv2dWgrad.
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv2d(conv::Operator convolutional_operator,
            conv::Conv2dProblemSize problem_size,
            TensorRef<ElementA, LayoutA> tensor_A,
            TensorRef<ElementB, LayoutB> tensor_B,
            TensorRef<ElementC, LayoutC> tensor_C,
            TensorRef<ElementC, LayoutC> tensor_D, ElementCompute alpha,
            ElementCompute beta) {
    switch (convolutional_operator) {
        case conv::Operator::kFprop:
            Conv2dFprop<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        case conv::Operator::kDgrad:
            Conv2dDgrad<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        case conv::Operator::kWgrad:
            Conv2dWgrad<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        default:
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 3D convolution
////////////////////////////////////////////////////////////////////////////////////////////////////

/// y = conv3d(x, w)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv3dFprop(conv::Conv3dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_x,
                 TensorRef<ElementB, LayoutB> tensor_w,
                 TensorRef<ElementC, LayoutC> tensor_y_in,
                 TensorRef<ElementC, LayoutC> tensor_y_out,
                 ElementCompute alpha, ElementCompute beta) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < problem_size.N; ++n) {
        for (int z = 0; z < problem_size.Z; ++z) {
            for (int p = 0; p < problem_size.P; ++p) {
                for (int q = 0; q < problem_size.Q; ++q) {
                    for (int k = 0; k < problem_size.K; ++k) {
                        ElementAccumulator acc = ElementAccumulator();

                        for (int t = 0; t < problem_size.T; ++t) {
                            for (int r = 0; r < problem_size.R; ++r) {
                                for (int s = 0; s < problem_size.S; ++s) {
                                    for (int c = 0; c < problem_size.C; ++c) {
                                        int filter_t = t;
                                        int filter_r = r;
                                        int filter_s = s;

                                        if (problem_size.mode ==
                                            cutlass::conv::Mode::kConvolution) {
                                            filter_t = problem_size.T - 1 - t;
                                            filter_r = problem_size.R - 1 - r;
                                            filter_s = problem_size.S - 1 - s;
                                        }

                                        int d = z * problem_size.stride_d -
                                                problem_size.pad_d +
                                                filter_t *
                                                        problem_size.dilation_d;
                                        int h = p * problem_size.stride_h -
                                                problem_size.pad_h +
                                                filter_r *
                                                        problem_size.dilation_h;
                                        int w = q * problem_size.stride_w -
                                                problem_size.pad_w +
                                                filter_s *
                                                        problem_size.dilation_w;

                                        if (d >= 0 && d < problem_size.D &&
                                            h >= 0 && h < problem_size.H &&
                                            w >= 0 && w < problem_size.W) {
                                            ElementA a = tensor_x.at(
                                                    {n, d, h, w, c});
                                            ElementB b = tensor_w.at(
                                                    {k, t, r, s, c});

                                            acc = inner_product_op(
                                                    ElementAccumulator(a),
                                                    ElementAccumulator(b), acc);
                                        }
                                    }
                                }
                            }
                        }

                        // Apply Epilogue, compute ElementCompute,
                        // convert and store ElementC
                        ElementC c_ref = ElementC();

                        if (beta != ElementCompute()) {
                            c_ref = tensor_y_in.at(
                                    cutlass::make_Coord(n, z, p, q, k));
                        }

                        tensor_y_out.at(cutlass::make_Coord(n, z, p, q, k)) =
                                convert_op(alpha * ElementCompute(acc) +
                                           beta * ElementCompute(c_ref));
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Dgrad
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dx = dgrad(dy, w)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv3dDgrad(cutlass::conv::Conv3dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_dy,
                 TensorRef<ElementB, LayoutB> tensor_w,
                 TensorRef<ElementC, LayoutC> tensor_dx_in,
                 TensorRef<ElementC, LayoutC> tensor_dx_out,
                 ElementCompute alpha, ElementCompute beta) {
    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int n = 0; n < problem_size.N; ++n) {
        for (int d = 0; d < problem_size.D; ++d) {
            for (int h = 0; h < problem_size.H; ++h) {
                for (int w = 0; w < problem_size.W; ++w) {
                    for (int c = 0; c < problem_size.C; ++c) {
                        ElementAccumulator acc = ElementAccumulator();

                        for (int t = 0; t < problem_size.T; ++t) {
                            for (int r = 0; r < problem_size.R; ++r) {
                                for (int s = 0; s < problem_size.S; ++s) {
                                    for (int k = 0; k < problem_size.K; ++k) {
                                        int filter_t = t;
                                        int filter_r = r;
                                        int filter_s = s;

                                        if (problem_size.mode ==
                                            cutlass::conv::Mode::kConvolution) {
                                            filter_t = problem_size.T - 1 - t;
                                            filter_r = problem_size.R - 1 - r;
                                            filter_s = problem_size.S - 1 - s;
                                        }

                                        int z = d + problem_size.pad_d -
                                                filter_t *
                                                        problem_size.dilation_d;
                                        int p = h + problem_size.pad_h -
                                                filter_r *
                                                        problem_size.dilation_h;
                                        int q = w + problem_size.pad_w -
                                                filter_s *
                                                        problem_size.dilation_w;

                                        if (z >= 0 &&
                                            (z % problem_size.stride_d) == 0 &&
                                            p >= 0 &&
                                            (p % problem_size.stride_h) == 0 &&
                                            q >= 0 &&
                                            (q % problem_size.stride_w) == 0) {
                                            z = z / problem_size.stride_d;
                                            p = p / problem_size.stride_h;
                                            q = q / problem_size.stride_w;

                                            if (z < problem_size.Z &&
                                                p < problem_size.P &&
                                                q < problem_size.Q) {
                                                ElementA a = tensor_dy.at(
                                                        cutlass::make_Coord(
                                                                n, z, p, q, k));
                                                ElementB b = tensor_w.at(
                                                        cutlass::make_Coord(
                                                                k, t, r, s, c));

                                                acc = inner_product_op(
                                                        ElementAccumulator(a),
                                                        ElementAccumulator(b),
                                                        acc);
                                            }
                                        }

                                    }  // for (K)
                                }      // for (S)
                            }          // for (R)
                        }              // for (T)

                        // Apply Epilogue, compute ElementCompute,
                        // convert and store ElementC
                        ElementC c_ref = ElementC();

                        if (beta != ElementCompute()) {
                            c_ref = tensor_dx_in.at(
                                    cutlass::make_Coord(n, d, h, w, c));
                        }

                        tensor_dx_out.at(cutlass::make_Coord(n, d, h, w, c)) =
                                convert_op(alpha * ElementCompute(acc) +
                                           beta * ElementCompute(c_ref));

                    }  // for (C)
                }      // for (W)
            }          // for (H)
        }              // for (D)
    }                  // for (N)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wgrad
////////////////////////////////////////////////////////////////////////////////////////////////////

/// dw = wgrad(dy, x)
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv3dWgrad(cutlass::conv::Conv3dProblemSize problem_size,
                 TensorRef<ElementA, LayoutA> tensor_dy,
                 TensorRef<ElementB, LayoutB> tensor_x,
                 TensorRef<ElementC, LayoutC> tensor_dw_in,
                 TensorRef<ElementC, LayoutC> tensor_dw_out,
                 ElementCompute alpha, ElementCompute beta) {
    InnerProductOp inner_product_op;
    ConvertOp convert_op;

    // Apply MMA and accumulate ElementAccumulator
    for (int k = 0; k < problem_size.K; ++k) {
        for (int t = 0; t < problem_size.T; ++t) {
            for (int r = 0; r < problem_size.R; ++r) {
                for (int s = 0; s < problem_size.S; ++s) {
                    for (int c = 0; c < problem_size.C; ++c) {
                        ElementAccumulator acc = ElementAccumulator();

                        for (int n = 0; n < problem_size.N; ++n) {
                            for (int z = 0; z < problem_size.Z; ++z) {
                                for (int p = 0; p < problem_size.P; ++p) {
                                    for (int q = 0; q < problem_size.Q; ++q) {
                                        int filter_t = t;
                                        int filter_r = r;
                                        int filter_s = s;

                                        if (problem_size.mode ==
                                            cutlass::conv::Mode::kConvolution) {
                                            filter_t = problem_size.T - 1 - t;
                                            filter_r = problem_size.R - 1 - r;
                                            filter_s = problem_size.S - 1 - s;
                                        }

                                        Tensor5DCoord b_coord = make_Coord(
                                                n,
                                                z * problem_size.stride_d -
                                                        problem_size.pad_d +
                                                        filter_t *
                                                                problem_size
                                                                        .dilation_d,
                                                p * problem_size.stride_h -
                                                        problem_size.pad_h +
                                                        filter_r *
                                                                problem_size
                                                                        .dilation_h,
                                                q * problem_size.stride_w -
                                                        problem_size.pad_w +
                                                        filter_s *
                                                                problem_size
                                                                        .dilation_w,
                                                c);

                                        if (b_coord.d() < problem_size.D &&
                                            b_coord.d() >= 0 &&
                                            b_coord.h() < problem_size.H &&
                                            b_coord.h() >= 0 &&
                                            b_coord.w() < problem_size.W &&
                                            b_coord.w() >= 0) {
                                            ElementAccumulator a =
                                                    ElementAccumulator(tensor_dy.at(
                                                            cutlass::make_Coord(
                                                                    n, z, p, q,
                                                                    k)));
                                            ElementAccumulator b =
                                                    ElementAccumulator(
                                                            tensor_x.at(
                                                                    b_coord));

                                            acc = inner_product_op(a, b, acc);
                                        }
                                    }
                                }
                            }
                        }

                        // Apply Epilogue, compute ElementCompute,
                        // convert and store ElementC
                        ElementC c_ref = ElementC();

                        if (beta != ElementCompute()) {
                            c_ref = tensor_dw_in.at(
                                    cutlass::make_Coord(k, t, r, s, c));
                        }

                        tensor_dw_out.at(cutlass::make_Coord(k, t, r, s, c)) =
                                convert_op(alpha * ElementCompute(acc) +
                                           beta * ElementCompute(c_ref));

                    }  // for (C)
                }      // for (S)
            }          // for (R)
        }              // for (T)
    }                  // for (K)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Generic 3D convolution targeting Conv2dFprop, Conv2dDgrad, and
/// Conv2dWgrad.
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ElementCompute, typename ElementAccumulator = ElementCompute,
          typename ConvertOp = NumericConverter<ElementC, ElementCompute>,
          typename InnerProductOp = multiply_add<ElementAccumulator>>
void Conv3d(conv::Operator convolutional_operator,
            conv::Conv3dProblemSize problem_size,
            TensorRef<ElementA, LayoutA> tensor_A,
            TensorRef<ElementB, LayoutB> tensor_B,
            TensorRef<ElementC, LayoutC> tensor_C,
            TensorRef<ElementC, LayoutC> tensor_D, ElementCompute alpha,
            ElementCompute beta) {
    switch (convolutional_operator) {
        case conv::Operator::kFprop:
            Conv3dFprop<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        case conv::Operator::kDgrad:
            Conv3dDgrad<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        case conv::Operator::kWgrad:
            Conv3dWgrad<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                        ElementCompute, ElementAccumulator, ConvertOp,
                        InnerProductOp>(problem_size, tensor_A, tensor_B,
                                        tensor_C, tensor_D, alpha, beta);
            break;

        default:
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general convolution among tensors of rank=4 pointed
/// to by TensorRef objects.
template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_convolution(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                         TensorRef<ElementSrc, LayoutSrc> tensor_src,
                         TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                         ScalarType beta,
                         TensorRef<ElementBias, LayoutBias> tensor_bias,
                         ScalarType gamma,
                         TensorRef<ElementDst, LayoutDst> tensor_z,
                         TensorRef<ElementDst, LayoutDst> tensor_dst,
                         ComputeType initial_accum) {
    static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                          LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                  "Tensors must be of rank 4");
    using TensorCoordSrc = typename LayoutSrc::TensorCoord;
    using TensorCoordFilter = typename LayoutFilter::TensorCoord;
    using TensorCoordBias = typename LayoutBias::TensorCoord;
    using TensorCoordDst = typename LayoutDst::TensorCoord;

    int const N = conv_param.N;
    int const IC = conv_param.C;
    int const OC = conv_param.K;
    int const IH = conv_param.H;
    int const IW = conv_param.W;
    int const OH = conv_param.P;
    int const OW = conv_param.Q;
    int const FH = conv_param.R;
    int const FW = conv_param.S;
    int const PH = conv_param.pad_h;
    int const PW = conv_param.pad_w;
    int const SH = conv_param.stride_h;
    int const SW = conv_param.stride_w;

    // Blocking necessary to speedup reference implementation
    int const Mblock = 16;
    int const Nblock = 16;

    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    for (int n_block = 0; n_block < N; n_block += Nblock) {
        for (int oc_block = 0; oc_block < OC; oc_block += Mblock) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    ComputeType accum[Mblock][Nblock];

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            accum[i][j] = initial_accum;
                        }
                    }

                    int ih_base = oh * SH - PH;
                    int iw_base = ow * SW - PW;

                    for (int fh = 0; fh < FH; fh++) {
                        for (int fw = 0; fw < FW; fw++) {
                            for (int ic = 0; ic < IC; ic++) {
                                for (int j = 0; j < Nblock; j++) {
                                    for (int i = 0; i < Mblock; i++) {
                                        int n = n_block + j;
                                        int oc = oc_block + i;

                                        int ih = ih_base + fh;
                                        int iw = iw_base + fw;
                                        if (n < N && oc < OC) {
                                            ElementSrc src;
                                            if (ih >= 0 && ih < IH && iw >= 0 &&
                                                iw < IW) {
                                                src = tensor_src.at(
                                                        TensorCoordSrc(n, ih,
                                                                       iw, ic));
                                            } else {
                                                src = 0;
                                            }
                                            ElementFilter filter =
                                                    tensor_filter.at(
                                                            TensorCoordFilter(
                                                                    oc, fh, fw,
                                                                    ic));

                                            ComputeType compute_src(
                                                    cast_if_scalar<ComputeType>(
                                                            src));
                                            ComputeType compute_filter(
                                                    cast_if_scalar<ComputeType>(
                                                            filter));

                                            accum[i][j] = inner_product_op(
                                                    compute_filter, compute_src,
                                                    accum[i][j]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            int n = n_block + j;
                            int oc = oc_block + i;

                            TensorCoordDst coord(n, oh, ow, oc);
                            TensorCoordBias coord_bias(0, 0, 0, oc);

                            if (n < N && oc < OC) {
                                ScalarType intermediate =
                                        alpha * ScalarType(accum[i][j]) +
                                        beta * ScalarType(tensor_bias.at(
                                                       coord_bias)) +
                                        gamma * ScalarType(tensor_z.at(coord));
                                if (detail::need_round<ElementDst,
                                                       ScalarType>::value) {
                                    intermediate = std::round(intermediate);
                                }
                                tensor_dst.at(coord) = convert_op(intermediate);
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2)
/// pointed to by TensorRef objects.
template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_convolution(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                         TensorRef<ElementSrc, LayoutSrc> tensor_src,
                         TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                         ScalarType beta,
                         TensorRef<ElementBias, LayoutBias> tensor_bias,
                         TensorRef<ElementDst, LayoutDst> tensor_dst,
                         ComputeType initial_accum) {
    compute_convolution<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                        ElementDst, LayoutDst, ScalarType, ComputeType,
                        InnerProductOp, ConvertOp>(
            conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias, 0,
            tensor_dst, tensor_dst, initial_accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a batch convolution among source tensor of rank=4 and
/// filter tensor of rank=5 pointed to by TensorRef objects.
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_batch_convolution(
        conv::Conv2dProblemSize conv_param, ScalarType alpha,
        TensorRef<ElementSrc, LayoutSrc> tensor_src,
        TensorRef<ElementFilter, LayoutFilter> tensor_filter, ScalarType beta,
        TensorRef<ElementBias, LayoutBias> tensor_bias, ScalarType gamma,
        TensorRef<ElementDst, LayoutDst> tensor_z,
        TensorRef<ElementDst, LayoutDst> tensor_dst,
        ComputeType initial_accum) {
    static_assert(LayoutSrc::kRank == 4 && LayoutDst::kRank == 4 &&
                          LayoutBias::kRank == 4,
                  "Tensors must be of rank 4");
    static_assert(LayoutFilter::kRank == 5, "Filter must be of rank 5");
    using TensorCoordSrc = typename LayoutSrc::TensorCoord;
    using TensorCoordFilter = typename LayoutFilter::TensorCoord;
    using TensorCoordBias = typename LayoutBias::TensorCoord;
    using TensorCoordDst = typename LayoutDst::TensorCoord;

    int const N = conv_param.N;
    int const IC = conv_param.C;
    int const OC = conv_param.K;
    int const IH = conv_param.H;
    int const IW = conv_param.W;
    int const OH = conv_param.P;
    int const OW = conv_param.Q;
    int const FH = conv_param.R;
    int const FW = conv_param.S;
    int const PH = conv_param.pad_h;
    int const PW = conv_param.pad_w;
    int const SH = conv_param.stride_h;
    int const SW = conv_param.stride_w;

    // Blocking necessary to speedup reference implementation
    int const Mblock = 16;
    int const Nblock = 16;

    ConvertOp convert_op;
    InnerProductOp inner_product_op;

    for (int n_block = 0; n_block < N; n_block += Nblock) {
        for (int oc_block = 0; oc_block < OC; oc_block += Mblock) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    ComputeType accum[Mblock][Nblock];

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            accum[i][j] = initial_accum;
                        }
                    }

                    int ih_base = oh * SH - PH;
                    int iw_base = ow * SW - PW;

                    for (int fh = 0; fh < FH; fh++) {
                        for (int fw = 0; fw < FW; fw++) {
                            for (int ic = 0; ic < IC; ic++) {
                                for (int j = 0; j < Nblock; j++) {
                                    for (int i = 0; i < Mblock; i++) {
                                        int n = n_block + j;
                                        int oc = oc_block + i;

                                        int ih = ih_base + fh;
                                        int iw = iw_base + fw;
                                        if (n < N && oc < OC) {
                                            ElementSrc src;
                                            if (ih >= 0 && ih < IH && iw >= 0 &&
                                                iw < IW) {
                                                src = tensor_src.at(
                                                        TensorCoordSrc(n, ih,
                                                                       iw, ic));
                                            } else {
                                                src = 0;
                                            }
                                            ElementFilter filter =
                                                    tensor_filter.at(
                                                            TensorCoordFilter(
                                                                    n, oc, fh,
                                                                    fw, ic));

                                            ComputeType compute_src(
                                                    cast_if_scalar<ComputeType>(
                                                            src));
                                            ComputeType compute_filter(
                                                    cast_if_scalar<ComputeType>(
                                                            filter));

                                            accum[i][j] = inner_product_op(
                                                    compute_filter, compute_src,
                                                    accum[i][j]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int j = 0; j < Nblock; j++) {
                        for (int i = 0; i < Mblock; i++) {
                            int n = n_block + j;
                            int oc = oc_block + i;

                            TensorCoordDst coord(n, oh, ow, oc);
                            TensorCoordBias coord_bias(0, 0, 0, oc);

                            if (n < N && oc < OC) {
                                ScalarType intermediate =
                                        alpha * ScalarType(accum[i][j]) +
                                        beta * ScalarType(tensor_bias.at(
                                                       coord_bias)) +
                                        gamma * ScalarType(tensor_z.at(coord));
                                if (detail::need_round<ElementDst,
                                                       ScalarType>::value) {
                                    intermediate = std::round(intermediate);
                                }
                                tensor_dst.at(coord) = convert_op(intermediate);
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a batch convolution among source tensor of rank=4 and
/// filter tensor of rank=5 pointed to by TensorRef objects.
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverter<ElementDst, ScalarType>>
void compute_batch_convolution(
        conv::Conv2dProblemSize conv_param, ScalarType alpha,
        TensorRef<ElementSrc, LayoutSrc> tensor_src,
        TensorRef<ElementFilter, LayoutFilter> tensor_filter, ScalarType beta,
        TensorRef<ElementBias, LayoutBias> tensor_bias,
        TensorRef<ElementDst, LayoutDst> tensor_dst,
        ComputeType initial_accum) {
    compute_batch_convolution<ElementSrc, LayoutSrc, ElementFilter,
                              LayoutFilter, ElementDst, LayoutDst, ScalarType,
                              ComputeType, InnerProductOp, ConvertOp>(
            conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias, 0,
            tensor_dst, tensor_dst, initial_accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = cutlass::arch::OpMultiplyAdd>
struct Convolution;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Normal convolution speicialization
template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp>
struct Convolution {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            multiply_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for XOR-popc
template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType>
struct Convolution<ConvolutionType, ElementSrc, LayoutSrc, ElementFilter,
                   LayoutFilter, ElementDst, LayoutDst, ElementBias, LayoutBias,
                   ScalarType, ComputeType, arch::OpXorPopc> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            xor_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        compute_convolution<ConvolutionType, ElementSrc, LayoutSrc,
                            ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                            ElementBias, LayoutBias, ScalarType, ComputeType,
                            xor_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Batch convolution speicialization
/// Partial specialization for multiply-add
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType, typename InnerProductOp>
struct Convolution<conv::ConvType::kBatchConvolution, ElementSrc, LayoutSrc,
                   ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                   ElementBias, LayoutBias, ScalarType, ComputeType,
                   InnerProductOp> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutDst::kRank == 4 &&
                              LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        static_assert(LayoutFilter::kRank == 5, "Filter must be of rank 5");

        compute_batch_convolution<
                ElementSrc, LayoutSrc, ElementFilter, LayoutFilter, ElementDst,
                LayoutDst, ElementBias, LayoutBias, ScalarType, ComputeType,
                multiply_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutDst::kRank == 4 &&
                              LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        static_assert(LayoutFilter::kRank == 5, "Filter must be of rank 5");

        compute_batch_convolution<
                ElementSrc, LayoutSrc, ElementFilter, LayoutFilter, ElementDst,
                LayoutDst, ElementBias, LayoutBias, ScalarType, ComputeType,
                multiply_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Parital specialization for XOR-popc
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType>
struct Convolution<conv::ConvType::kBatchConvolution, ElementSrc, LayoutSrc,
                   ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                   ElementBias, LayoutBias, ScalarType, ComputeType,
                   arch::OpXorPopc> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutDst::kRank == 4 &&
                              LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        static_assert(LayoutFilter::kRank == 5, "Filter must be of rank 5");

        compute_batch_convolution<ElementSrc, LayoutSrc, ElementFilter,
                                  LayoutFilter, ElementDst, LayoutDst,
                                  ElementBias, LayoutBias, ScalarType,
                                  ComputeType, xor_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                tensor_dst, initial_accum);
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutDst::kRank == 4 &&
                              LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        static_assert(LayoutFilter::kRank == 5, "Filter must be of rank 5");

        compute_batch_convolution<ElementSrc, LayoutSrc, ElementFilter,
                                  LayoutFilter, ElementDst, LayoutDst,
                                  ElementBias, LayoutBias, ScalarType,
                                  ComputeType, xor_add<ComputeType>, ConvertOp>(
                conv_param, alpha, tensor_src, tensor_filter, beta, tensor_bias,
                gamma, tensor_z, tensor_dst, initial_accum);
    }
};

template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType,
          typename InnerProductOp = multiply_add<ComputeType>,
          typename ConvertOp = NumericConverterClamp<ElementDst, ScalarType>>
struct Deconvolution {
    void operator()(conv::Conv2dProblemSize problem_size, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        ConvertOp convert_op;
        InnerProductOp inner_product_op;

        // Apply MMA and accumulate ComputeType
        for (int n = 0; n < problem_size.N; ++n) {
            for (int h = 0; h < problem_size.H; ++h) {
                for (int w = 0; w < problem_size.W; ++w) {
                    for (int c = 0; c < problem_size.C; ++c) {
                        ComputeType acc = initial_accum;

                        for (int r = 0; r < problem_size.R; ++r) {
                            for (int s = 0; s < problem_size.S; ++s) {
                                for (int k = 0; k < problem_size.K; ++k) {
                                    int filter_r = r;
                                    int filter_s = s;

                                    if (problem_size.mode ==
                                        cutlass::conv::Mode::kConvolution) {
                                        filter_r = problem_size.R - 1 - r;
                                        filter_s = problem_size.S - 1 - s;
                                    }

                                    int p = h + problem_size.pad_h -
                                            filter_r * problem_size.dilation_h;
                                    int q = w + problem_size.pad_w -
                                            filter_s * problem_size.dilation_w;

                                    if (p >= 0 &&
                                        (p % problem_size.stride_h) == 0 &&
                                        q >= 0 &&
                                        (q % problem_size.stride_w) == 0) {
                                        p = p / problem_size.stride_h;
                                        q = q / problem_size.stride_w;

                                        if (p < problem_size.P &&
                                            q < problem_size.Q) {
                                            ElementSrc a = tensor_src.at(
                                                    cutlass::make_Coord(n, p, q,
                                                                        k));
                                            ElementFilter b = tensor_filter.at(
                                                    cutlass::make_Coord(k, r, s,
                                                                        c));

                                            acc = inner_product_op(
                                                    ComputeType(a),
                                                    ComputeType(b), acc);
                                        }
                                    }

                                }  // for (K)
                            }      // for (S)
                        }          // for (R)

                        ElementBias bias_ref = ElementBias();

                        if (beta != ScalarType()) {
                            bias_ref = tensor_bias.at(
                                    cutlass::make_Coord(0, 0, 0, c));
                        }
                        ElementDst c_ref = ElementDst();

                        if (gamma != ScalarType()) {
                            c_ref = tensor_z.at(
                                    cutlass::make_Coord(n, h, w, c));
                        }

                        ScalarType intermediate = alpha * ScalarType(acc) +
                                                  beta * ScalarType(bias_ref) +
                                                  gamma * ScalarType(c_ref);
                        if (detail::need_round<ElementDst, ScalarType>::value) {
                            intermediate = std::round(intermediate);
                        }
                        tensor_dst.at(cutlass::make_Coord(n, h, w, c)) =
                                convert_op(intermediate);

                    }  // for (C)
                }      // for (W)
            }          // for (H)
        }              // for (N)
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// TODO: support bias and z
/// specialization for depthwise convolution
template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType, typename InnerProductOp>
struct Convolution<conv::ConvType::kDepthwiseConvolution, ElementSrc, LayoutSrc,
                   ElementFilter, LayoutFilter, ElementDst, LayoutDst,
                   ElementBias, LayoutBias, ScalarType, ComputeType,
                   InnerProductOp> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        Depsep_Fprop<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                     ElementBias, LayoutBias, ElementDst, LayoutDst, ScalarType,
                     ComputeType, ConvertOp, multiply_add<ComputeType>>(
                conv_param, tensor_src, tensor_filter, tensor_bias, tensor_dst,
                tensor_dst, alpha, beta, 0);
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        Depsep_Fprop<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                     ElementBias, LayoutBias, ElementDst, LayoutDst, ScalarType,
                     ComputeType, ConvertOp, multiply_add<ComputeType>>(
                conv_param, tensor_src, tensor_filter, tensor_bias, tensor_z,
                tensor_dst, alpha, beta, gamma);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementFilter, typename LayoutFilter,
          typename ElementDst, typename LayoutDst, typename ElementBias,
          typename LayoutBias, typename ScalarType, typename ComputeType,
          typename InnerProductOp = cutlass::arch::OpMultiplyAdd>
struct Deconv2d;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementSrc, typename LayoutSrc, typename ElementFilter,
          typename LayoutFilter, typename ElementDst, typename LayoutDst,
          typename ElementBias, typename LayoutBias, typename ScalarType,
          typename ComputeType, typename InnerProductOp>
struct Deconv2d<conv::ConvType::kDepthwiseConvolution, ElementSrc, LayoutSrc,
                ElementFilter, LayoutFilter, ElementDst, LayoutDst, ElementBias,
                LayoutBias, ScalarType, ComputeType, InnerProductOp> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementDst>::value,
            NumericConverterClamp<ElementDst, ScalarType>,
            NumericConverter<ElementDst, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");
        Depsep_Dgrad<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                     ElementBias, LayoutBias, ElementDst, LayoutDst, ScalarType,
                     ComputeType, ConvertOp, multiply_add<ComputeType>>(
                conv_param, tensor_src, tensor_filter, tensor_bias, tensor_dst,
                tensor_dst, alpha, beta, ScalarType(0));
    }

    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementFilter, LayoutFilter> tensor_filter,
                    ScalarType beta,
                    TensorRef<ElementBias, LayoutBias> tensor_bias,
                    ScalarType gamma, TensorRef<ElementDst, LayoutDst> tensor_z,
                    TensorRef<ElementDst, LayoutDst> tensor_dst,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutFilter::kRank == 4 &&
                              LayoutDst::kRank == 4 && LayoutBias::kRank == 4,
                      "Tensors must be of rank 4");

        Depsep_Dgrad<ElementSrc, LayoutSrc, ElementFilter, LayoutFilter,
                     ElementBias, LayoutBias, ElementDst, LayoutDst, ScalarType,
                     ComputeType, ConvertOp, multiply_add<ComputeType>>(
                conv_param, tensor_src, tensor_filter, tensor_bias, tensor_z,
                tensor_dst, alpha, beta, gamma);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <conv::ConvType ConvolutionType, typename ElementSrc,
          typename LayoutSrc, typename ElementDiff, typename LayoutDiff,
          typename ElementGrad, typename LayoutGrad, typename ScalarType,
          typename ComputeType,
          typename InnerProductOp = cutlass::arch::OpMultiplyAdd>
struct Convolution2dBackwardFilter;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ElementSrc, typename LayoutSrc, typename ElementDiff,
          typename LayoutDiff, typename ElementGrad, typename LayoutGrad,
          typename ScalarType, typename ComputeType, typename InnerProductOp>
struct Convolution2dBackwardFilter<conv::ConvType::kDepthwiseConvolution,
                                   ElementSrc, LayoutSrc, ElementDiff,
                                   LayoutDiff, ElementGrad, LayoutGrad,
                                   ScalarType, ComputeType, InnerProductOp> {
    using ConvertOp = typename platform::conditional<
            detail::need_clamp<ElementGrad>::value,
            NumericConverterClamp<ElementGrad, ScalarType>,
            NumericConverter<ElementGrad, ScalarType>>::type;
    void operator()(conv::Conv2dProblemSize conv_param, ScalarType alpha,
                    TensorRef<ElementSrc, LayoutSrc> tensor_src,
                    TensorRef<ElementDiff, LayoutDiff> tensor_diff,
                    TensorRef<ElementGrad, LayoutGrad> tensor_grad,
                    ComputeType initial_accum = ComputeType(0)) {
        static_assert(LayoutSrc::kRank == 4 && LayoutDiff::kRank == 4 &&
                              LayoutGrad::kRank == 4,
                      "Tensors must be of rank 4");
        Depsep_Wgrad<ElementSrc, LayoutSrc, ElementDiff, LayoutDiff,
                     ElementGrad, LayoutGrad, ScalarType, ComputeType,
                     ConvertOp, multiply_add<ComputeType>>(
                conv_param, tensor_src, tensor_diff, tensor_grad, alpha);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace host
}  // namespace reference
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
