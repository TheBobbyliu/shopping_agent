async (page) => {
  const appUrl = process.env.FRONTEND_URL || "http://127.0.0.1:3000";
  const uploadPath = process.env.UPLOAD_FIXTURE_PATH;

  if (!uploadPath) {
    throw new Error("UPLOAD_FIXTURE_PATH is required");
  }

  const sseBody = [
    'event: status',
    'data: {"text":"Searching for products..."}',
    "",
    'event: token',
    'data: {"text":"Image reply"}',
    "",
    'event: done',
    'data: {"reply":"Image reply","session_id":"sess_image_history_test","tool_calls":[]}',
    "",
  ].join("\n");

  await page.addInitScript((mockSseBody) => {
    localStorage.clear();

    const originalFetch = window.fetch.bind(window);
    const encoder = new TextEncoder();

    window.fetch = async (input, init) => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof Request
            ? input.url
            : String(input);

      if (url.endsWith("/api/ready")) {
        return new Response(JSON.stringify({ status: "ready" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      if (url.endsWith("/api/chat/stream")) {
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(encoder.encode(mockSseBody));
              controller.close();
            },
          }),
          {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          }
        );
      }

      return originalFetch(input, init);
    };
  }, sseBody);

  await page.goto(appUrl);
  await page.waitForSelector("text=What are you looking for?");

  await page.locator('input[type="file"]').setInputFiles(uploadPath);
  await page.locator("textarea").locator("xpath=following-sibling::button[1]").click();

  await page.waitForFunction(() => {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    if (!sessions.length) return false;

    const key = `messages_${sessions[0].session_id}`;
    const messages = JSON.parse(localStorage.getItem(key) || "[]");

    return messages.some(
      (message) =>
        message.role === "user" &&
        typeof message.imagePreview === "string" &&
        message.imagePreview.startsWith("data:image/")
    );
  });

  await page.getByRole("button", { name: "New chat" }).click();
  await page.waitForSelector("text=What are you looking for?");
  await page.locator("aside").getByText("Image search").first().click();

  await page.waitForFunction(() => {
    const preview = document.querySelector('img[alt="upload"]');
    return Boolean(preview && preview.getAttribute("src")?.startsWith("data:image/"));
  });

  const result = await page.evaluate(() => {
    const sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
    const key = `messages_${sessions[0].session_id}`;
    const messages = JSON.parse(localStorage.getItem(key) || "[]");
    const userMessage = messages.find((message) => message.role === "user");
    const visiblePreview = document
      .querySelector('img[alt="upload"]')
      ?.getAttribute("src")
      ?.startsWith("data:image/");

    return {
      sessionCount: sessions.length,
      storedPreview:
        typeof userMessage?.imagePreview === "string" &&
        userMessage.imagePreview.startsWith("data:image/"),
      visiblePreview: Boolean(visiblePreview),
    };
  });

  if (!result.storedPreview || !result.visiblePreview) {
    throw new Error(`Image preview was not preserved: ${JSON.stringify(result)}`);
  }

  console.log(JSON.stringify(result));
}
