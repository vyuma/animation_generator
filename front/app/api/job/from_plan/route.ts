import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

/**
 * 承認済みプランから動画生成ジョブを作成する
 * POST /api/job/animation/generate
 *
 * レンダリングは数分かかるため同期 API (/api/animation) ではなくジョブ API を使い、
 * job_id を返す（進捗は /api/job/{job_id}/stream で受け取る）。
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${API_URL}/api/job/animation/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        generation_id: body.generation_id,
        content: body.content,
        enhance_prompt: body.enhance_prompt || "",
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: error || "動画生成ジョブの作成に失敗しました" },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      ok: true,
      job_id: data.job_id,
      generation_id: body.generation_id,
      message: data.message,
    });
  } catch (error) {
    console.error("Generate from plan error:", error);
    return NextResponse.json(
      { error: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
