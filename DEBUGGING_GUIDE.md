# 🔧 Debugging Guide: Serverless AI Image Editing Application

## Architecture Overview

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────────┐
│  React SPA   │────▶│ API Gateway   │────▶│ AWS Lambda   │────▶│ Amazon Bedrock   │
│  (Amplify)   │◀────│ (REST API)    │◀────│ (Python 3.x) │◀────│ Titan Image v2   │
└──────────────┘     └───────────────┘     └──────────────┘     └─────────────────┘
       │                                          │
       │                                          ▼
       │                                   ┌─────────────┐
       │                                   │  DynamoDB    │
       │                                   │  (optional)  │
       │                                   └─────────────┘
       ▼
  Amazon Cognito
  (Authentication)
```

---

## Bug #1: `TypeError: Cannot read properties of undefined (reading 'length')`

### Root Cause

The frontend crashes when the API returns a response that doesn't contain an `images` array.

**Crash chain:**

1. Lambda encounters an error (Bedrock validation, timeout, permissions, etc.)
2. Lambda returns `{ "error": "...", "statusCode": 500 }` **without** an `images` field
3. Frontend calls `ie(te.images)` → `te.images` is `undefined`
4. `ResultsViewer` component tries `s.length` on `undefined` → **💥 CRASH**

### Where the crash occurs (in the minified bundle)

```
File: assets/index-7vDF1VVo.js

# Line ~104 — the API call site:
const te = await C.generateImages(Ee, a, $);
ie(te.images);     // ← te.images is undefined when API returns error
                   //   ie() sets state, component re-renders

# Line ~102 — the ResultsViewer component:
function lg({images: s, ...}) {
  v.current = v.current.slice(0, s.length);  // ← CRASH: s is undefined
  ...
  s.length === 0 ? ...   // ← Also crashes
  s.map(...)             // ← Also crashes
}
```

### Fix Applied (Frontend)

Three layers of defense were added to the frontend bundle:

#### Layer 1: Safe response normalization in `generateImages()`
```js
// Before (UNSAFE):
const v = await d.json();
return v;

// After (SAFE):
const v = await d.json();
return {
  images: Array.isArray(v.images) ? v.images : [],
  error: v.error || null,
  warning: v.warning || null
};
```

#### Layer 2: Safe access at the call site
```js
// Before (UNSAFE):
ie(te.images);

// After (SAFE):
if (te && te.error) {
  ve(`API Error: ${te.error}`);
  return;
}
const Ke = te && Array.isArray(te.images) ? te.images : [];
if (Ke.length === 0) {
  ve("No images were generated.");
  return;
}
ie(Ke);
```

#### Layer 3: Defensive defaults in `ResultsViewer`
```js
// Before (UNSAFE):
function lg({images: s, ...}) {
  s.length ...

// After (SAFE):
function lg({images: s = [], ...}) {
  const _s = Array.isArray(s) ? s : [];
  _s.length ...
```

---

## Bug #2: Lambda Missing CORS Headers on Errors

### Root Cause

When the Lambda throws an unhandled exception, it may return a raw error without CORS headers. The browser then blocks the response entirely, and the frontend `fetch()` gets a network error instead of a parseable error JSON.

### Fix Applied (Lambda)

See `lambda/lambda_function.py`. Every code path — including the catch-all `except Exception` — uses `build_response()` which always includes CORS headers:

```python
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,...",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}

def build_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }

# EVERY error path includes images: [] as a safe fallback
except Exception as e:
    return build_response(500, {
        "error": "Internal server error",
        "code": "INTERNAL_ERROR",
        "images": [],  # ← Frontend will never crash
    })
```

---

## Bug #3: `prepare_titan_request` Schema Mismatch (Titan v2)

### Common Mistakes

| Mistake | Correct Titan v2 Schema |
|---------|------------------------|
| Using `taskType: "IMAGE_VARIATION"` | Use `"INPAINTING"` or `"OUTPAINTING"` |
| Putting prompt in `textToImageParams` | Use `inPaintingParams.text` |
| Sending `mask` at top level | Use `inPaintingParams.maskImage` |
| Sending data-URL (`data:image/png;base64,...`) | Strip prefix, send raw base64 only |
| Missing `imageGenerationConfig` | Required: `numberOfImages`, `height`, `width` |

### Correct Titan v2 INPAINTING Request

```json
{
  "taskType": "INPAINTING",
  "inPaintingParams": {
    "text": "a beautiful garden with flowers",
    "image": "<RAW base64 — no data:image prefix>",
    "maskImage": "<RAW base64 — no data:image prefix>"
  },
  "imageGenerationConfig": {
    "numberOfImages": 1,
    "height": 512,
    "width": 512,
    "cfgScale": 8.0
  }
}
```

### Correct Titan v2 OUTPAINTING Request

```json
{
  "taskType": "OUTPAINTING",
  "outPaintingParams": {
    "text": "extend the landscape",
    "image": "<RAW base64>",
    "maskImage": "<RAW base64>",
    "outPaintingMode": "DEFAULT"
  },
  "imageGenerationConfig": {
    "numberOfImages": 1,
    "height": 512,
    "width": 512,
    "cfgScale": 8.0
  }
}
```

### Expected Bedrock Response

```json
{
  "images": [
    "<base64_encoded_png_image>"
  ]
}
```

---

## Deployment Checklist

### Lambda

- [ ] Upload `lambda/lambda_function.py` to your Lambda function
- [ ] Set environment variables:
  - `BEDROCK_REGION` = `us-east-1` (or your region)
  - `MODEL_ID` = `amazon.titan-image-generator-v2:0`
  - `ALLOWED_ORIGIN` = your Amplify domain (or `*` for dev)
- [ ] Ensure Lambda IAM role has `bedrock:InvokeModel` permission
- [ ] Set Lambda timeout to at least **60 seconds** (image generation is slow)
- [ ] Set Lambda memory to at least **512 MB**

### API Gateway

- [ ] Enable CORS on the API Gateway resource
- [ ] Ensure the `OPTIONS` method returns proper CORS headers
- [ ] Verify the integration type is Lambda Proxy (`AWS_PROXY`)

### Frontend

- [ ] Verify `config/config.js` has the correct `api.invokeUrl`
- [ ] Deploy the patched `assets/index-7vDF1VVo.js` to Amplify

---

## Testing with curl

```bash
# Test the API directly (replace values):
curl -X POST "https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: YOUR_JWT_TOKEN" \
  -d '{
    "prompt": {
      "text": "a beautiful sunset over the ocean",
      "mode": "INPAINTING"
    },
    "base_image": "<base64_image>",
    "mask": "<base64_mask>"
  }' | jq .
```

### Expected success response:
```json
{
  "images": ["iVBORw0KGgo..."]
}
```

### Expected error response (still valid, won't crash frontend):
```json
{
  "error": "Bedrock API error: ...",
  "code": "ValidationException",
  "images": []
}
```

---

## Common Bedrock Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationException` | Invalid image dimensions, bad base64, wrong schema | Check image is ≤ 1408×1408, valid base64, correct taskType |
| `AccessDeniedException` | Lambda role can't call Bedrock | Add `bedrock:InvokeModel` to IAM policy |
| `ThrottlingException` | Too many requests | Add retry with exponential backoff |
| `ModelNotReadyException` | Model not available in region | Check [Bedrock model availability](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) |
| `ServiceUnavailableException` | Bedrock service issue | Retry after a few seconds |

---

## Files Modified

| File | Change |
|------|--------|
| `lambda/lambda_function.py` | **NEW** — Hardened Lambda handler with CORS, validation, correct Titan v2 schema |
| `assets/index-7vDF1VVo.js` | **PATCHED** — Safe access pattern for API responses, defensive defaults in ResultsViewer |
| `DEBUGGING_GUIDE.md` | **NEW** — This file |
