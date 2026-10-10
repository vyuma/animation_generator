import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ videoId: string }> }
) {
  const { videoId } = await params;

  try {
    const response = await fetch(`${API_URL}/api/animation/${videoId}`, {
      method: "GET",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "動画の取得に失敗しました" },
        { status: response.status }
      );
    }

    const blob = await response.blob();

    return new Response(blob, {
      headers: {
        "Content-Type": "video/mp4",
        "Content-Disposition": `inline; filename="video-${videoId}.mp4"`,
      },
    });
  } catch (error) {
    console.error("Video fetch error:", error);
    return NextResponse.json(
      { error: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
