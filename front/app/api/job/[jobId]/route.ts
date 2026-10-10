import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

export async function DELETE(
  _request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;

  try {
    const response = await fetch(`${API_URL}/api/job/${jobId}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: error || "ジョブの削除に失敗しました" },
        { status: response.status }
      );
    }

    return NextResponse.json({ ok: true, message: "Job deleted" });
  } catch (error) {
    console.error("Delete job error:", error);
    return NextResponse.json(
      { error: "サーバーエラーが発生しました" },
      { status: 500 }
    );
  }
}
