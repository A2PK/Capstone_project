// app/api/auth/proxy-file/route.ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url);
    const url = searchParams.get('url');

    if (!url) {
        return NextResponse.json({ error: 'Missing file URL' }, { status: 400 });
    }

    try {
        const originalResponse = await fetch(url);

        if (!originalResponse.ok) {
            throw new Error(`Failed to fetch resource (${originalResponse.status}): ${originalResponse.statusText}`);
        }
        const bodyStream = originalResponse.body;
        const contentType = originalResponse.headers.get('Content-Type');
        const headers = new Headers();

        // --- CORS Headers ---
        headers.set('Access-Control-Allow-Origin', '*'); // Hoặc origin cụ thể
        headers.set('Access-Control-Allow-Methods', 'GET');
        headers.set('Access-Control-Allow-Headers', 'Content-Type');
        if (contentType) {
            headers.set('Content-Type', contentType);
        } else {
            // Có thể đặt một giá trị mặc định hoặc bỏ qua nếu không chắc chắn
            // headers.set('Content-Type', 'application/octet-stream'); // Ví dụ fallback
        }
        return new Response(bodyStream, {
            status: 200,
            headers: headers,
        });

    } catch (error) {
        console.error('File Proxy Error:', error);
        let errorMessage = 'An unexpected error occurred.';
        if (error instanceof Error) {
            errorMessage = error.message;
        } else if (typeof error === 'string') {
            errorMessage = error;
        }
        return NextResponse.json(
            { error: 'Failed to proxy file.', details: errorMessage },
            { status: 500 }
        );
    }
}