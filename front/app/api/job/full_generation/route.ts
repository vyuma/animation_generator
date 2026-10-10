import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // バックエンドはクエリパラメータで受け取る
    const params = new URLSearchParams({
      content: body.text,
      additional_instructions: body.additional_instructions || "",
    });

    const response = await fetch(
      `${API_URL}/api/job/animation/full?${params.toString()}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: error || "生成ジョブの作成に失敗しました" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Full generation job error:", error);
    return NextResponse.json(
      { error: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
