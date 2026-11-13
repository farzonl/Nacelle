float4 VSMain(float2 Position : POSITION) : SV_Position {
  return float4(Position, 0, 1);
}

float4 PSMain(float4 Position : SV_Position) : SV_Target {
  float X = Position.x;
  float Y = Position.y;

  // ddx_fine
  float4 ValX = float4(X * X + X * Y, -2 * X + Y, Y, X / 2 + 2 * Y);
  float4 DdxFine = ddx_fine(ValX);
  float ExpectedX = 2 * ((uint)Position.x & ~1u) + 2 + Y;
  bool okDdxFine = (DdxFine.x == ExpectedX) && (DdxFine.y == -2) && (DdxFine.z == 0) && (DdxFine.w == 0.5);

  // ddx_coarse
  float4 ValXC = float4(2 * X, -2 * X + Y, Y, X / 2 + 2 * Y);
  float4 DdxCoarse = ddx_coarse(ValXC);
  bool okDdxCoarse = (DdxCoarse.x == 2) && (DdxCoarse.y == -2) && (DdxCoarse.z == 0) && (DdxCoarse.w == 0.5);

  // ddy_coarse
  float4 ValYC = float4(2 * Y, -2 * Y + X, X, Y / 2 + 2 * X);
  float4 DdyCoarse = ddy_coarse(ValYC);
  bool okDdyCoarse = (DdyCoarse.x == 2) && (DdyCoarse.y == -2) && (DdyCoarse.z == 0) && (DdyCoarse.w == 0.5);

  // ddy_fine
  float4 ValY = float4(Y * Y + X * Y, -2 * Y + X, X, Y / 2 + 2 * X);
  float4 DdyFine = ddy_fine(ValY);
  float ExpectedY = 2 * ((uint)Position.y & ~1u) + 2 + X;
  bool okDdyFine = (DdyFine.x == ExpectedY) && (DdyFine.y == -2) && (DdyFine.z == 0) && (DdyFine.w == 0.5);

  return float4(okDdxFine ? 1.0 : 0.0,
                okDdxCoarse ? 1.0 : 0.0,
                okDdyCoarse ? 1.0 : 0.0,
                okDdyFine ? 1.0 : 0.0);
}
