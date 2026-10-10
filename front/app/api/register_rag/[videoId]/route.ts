import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

/**
 * RAG用に動画を登録する
 * POST /api/register_rag/{video_id}
 *
 * バックエンドで行われる処理:
 * 1. manim_codeの内容をgemini2.5-flash-liteで要約
 * 2. 要約した内容を埋め込みベクトル化
 * 3. video_idと要約内容とともにRAGに登録
 */
export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ videoId: string }> }
) {
  const { videoId } = await params;

  try {
    const response = await fetch(`${API_URL}/api/register_rag/${videoId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: error || "RAG登録に失敗しました" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("RAG registration error:", error);
    return NextResponse.json(
      { error: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
