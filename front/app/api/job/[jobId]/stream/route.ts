import { NextRequest } from "next/server";

const API_URL = process.env.API_URL || "http://localhost:8888";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;

  try {
    const response = await fetch(`${API_URL}/api/job/${jobId}/stream`, {
      headers: {
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });

    if (!response.ok || !response.body) {
      return new Response(
        JSON.stringify({ error: "ストリームの接続に失敗しました" }),
        {
          status: response.status || 500,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    // SSEストリームをプロキシ
    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let isClosed = false;

        const safeClose = () => {
          if (!isClosed) {
            isClosed = true;
            try {
              controller.close();
            } catch {
              // Already closed
            }
          }
        };

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              safeClose();
              break;
            }

            if (!isClosed) {
              const text = decoder.decode(value, { stream: true });
              try {
                controller.enqueue(new TextEncoder().encode(text));
              } catch {
                // Controller closed by client
                isClosed = true;
                break;
              }
            }
          }
        } catch (error) {
          console.error("Stream error:", error);
          safeClose();
        } finally {
          reader.releaseLock();
        }
      },
      cancel() {
        // Client disconnected
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    console.error("SSE proxy error:", error);
    return new Response(
      JSON.stringify({ error: "サーバーエラーが発生しました" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
