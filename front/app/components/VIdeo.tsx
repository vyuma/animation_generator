import React from "react"; 
import DawnloadButton from "@/app/components/DownloadButton";


type VideoUrl = {
    videoUrl: string|null;
}

const VideoComponent = ({videoUrl}:VideoUrl) => {

        if (!videoUrl) {
            return (
                <div className="w-full h-[27rem] bg-slate-200/50 dark:bg-slate-800/50 border border-cyan-400/30 dark:border-cyan-500/20 shadow-lg rounded-xl transition-colors">
                    {/* なにも表示しないスペース */}
                </div>
            );
        }
        // 動画が設定されている場合
        return (
            <div>
                {/* ここはリスポンス対応する必要ある */}
                <video controls width="768" className="object-fill rounded-xl border border-cyan-400/40 dark:border-cyan-500/30 shadow-lg shadow-cyan-500/10" autoPlay muted>
                    <source src={videoUrl} type="video/mp4" />
                    お使いのブラウザは video タグをサポートしていません。
                </video>
            </div>
        );
}


export default VideoComponent;