%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.LinAlgMatrixC9M32N32U0S1 = type { i8* }
%dx.types.LinAlgMatrixC9M32N32U1S1 = type { i8* }
%dx.types.LinAlgMatrixC9M32N32U2S1 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32)
declare i32 @dx.op.groupId.i32(i32, i32)
declare %dx.types.LinAlgMatrixC9M32N32U2S1 @dx.op.linAlgFillMatrix.C9M32N32U2S1(i32, i1, float)
declare %dx.types.LinAlgMatrixC9M32N32U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M32N32U0S1(i32, %dx.types.Handle, i32, i32, i32, i32)
declare %dx.types.LinAlgMatrixC9M32N32U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M32N32U1S1(i32, %dx.types.Handle, i32, i32, i32, i32)
declare %dx.types.LinAlgMatrixC9M32N32U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.C9M32N32U2S1.C9M32N32U0S1.C9M32N32U1S1.C9M32N32U2S1(i32, %dx.types.LinAlgMatrixC9M32N32U0S1, %dx.types.LinAlgMatrixC9M32N32U1S1, %dx.types.LinAlgMatrixC9M32N32U2S1)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M32N32U2S1(i32, %dx.types.LinAlgMatrixC9M32N32U2S1, %dx.types.Handle, i32, i32, i32, i32)

define void @CSMain() {
entry:
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %srvB = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 1, i32 1, i1 false)
  %srvA = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
  %cbv = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)

  %dims = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %cbv, i32 0)
  %M = extractvalue %dx.types.CBufRet.i32 %dims, 0
  %N = extractvalue %dx.types.CBufRet.i32 %dims, 1
  %K = extractvalue %dx.types.CBufRet.i32 %dims, 2

  %gx = call i32 @dx.op.groupId.i32(i32 94, i32 0)
  %gy = call i32 @dx.op.groupId.i32(i32 94, i32 1)
  %colBase = mul i32 %gx, 32
  %rowBase = mul i32 %gy, 32
  %okR = icmp ult i32 %rowBase, %M
  %okC = icmp ult i32 %colBase, %N
  %ok = and i1 %okR, %okC
  br i1 %ok, label %body, label %exit

body:
  %acc0 = call %dx.types.LinAlgMatrixC9M32N32U2S1 @dx.op.linAlgFillMatrix.C9M32N32U2S1(i32 323, i1 false, float 0.0)
  %strideA = mul i32 %K, 4
  %strideB = mul i32 %N, 4
  %rowOffElem = mul i32 %rowBase, %K
  %hasK = icmp ult i32 0, %K
  br i1 %hasK, label %loop, label %store

loop:
  %k = phi i32 [ 0, %body ], [ %kNext, %loop ]
  %acc = phi %dx.types.LinAlgMatrixC9M32N32U2S1 [ %acc0, %body ], [ %accNext, %loop ]

  %aElemOff = add i32 %rowOffElem, %k
  %aByteOff = mul i32 %aElemOff, 4
  %matA = call %dx.types.LinAlgMatrixC9M32N32U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M32N32U0S1(i32 325, %dx.types.Handle %srvA, i32 %aByteOff, i32 %strideA, i32 0, i32 4)

  %bRowElem = mul i32 %k, %N
  %bElemOff = add i32 %bRowElem, %colBase
  %bByteOff = mul i32 %bElemOff, 4
  %matB = call %dx.types.LinAlgMatrixC9M32N32U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M32N32U1S1(i32 325, %dx.types.Handle %srvB, i32 %bByteOff, i32 %strideB, i32 0, i32 4)

  %accNext = call %dx.types.LinAlgMatrixC9M32N32U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.C9M32N32U2S1.C9M32N32U0S1.C9M32N32U1S1.C9M32N32U2S1(i32 322, %dx.types.LinAlgMatrixC9M32N32U0S1 %matA, %dx.types.LinAlgMatrixC9M32N32U1S1 %matB, %dx.types.LinAlgMatrixC9M32N32U2S1 %acc)
  %kNext = add i32 %k, 32
  %loopCond = icmp ult i32 %kNext, %K
  br i1 %loopCond, label %loop, label %store

store:
  %accFinal = phi %dx.types.LinAlgMatrixC9M32N32U2S1 [ %acc0, %body ], [ %accNext, %loop ]
  %cRowElem = mul i32 %rowBase, %N
  %cElemOff = add i32 %cRowElem, %colBase
  %cByteOff = mul i32 %cElemOff, 4
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M32N32U2S1(i32 331, %dx.types.LinAlgMatrixC9M32N32U2S1 %accFinal, %dx.types.Handle %uav, i32 %cByteOff, i32 %strideB, i32 0, i32 4)
  br label %exit

exit:
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!9}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{!2, !5, !7, null}
!2 = !{!3, !4}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!4 = !{i32 1, i32 0, !"", i32 0, i32 1, i32 1, i32 11, i32 0, null}
!5 = !{!6}
!6 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!7 = !{!8}
!8 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 16, null}
!9 = !{void ()* @CSMain, !"CSMain", null, !1, !10}
!10 = !{i32 4, !11}
!11 = !{i32 32, i32 1, i32 1}
