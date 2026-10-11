# Clerk RBAC setup

This project uses Clerk `public_metadata` as the role source of truth:

```json
{
  "role": "student",
  "department": "cs"
}
```

`role` must be `admin`, `faculty`, or `student`. Missing or invalid roles are treated as `student` by the backend.

## Environment

Copy the Clerk values into `.env`:

```env
CLERK_SECRET_KEY=sk_test_...
CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_JWKS_URL=https://YOUR_INSTANCE.clerk.accounts.dev/.well-known/jwks.json
CLERK_ISSUER=https://YOUR_INSTANCE.clerk.accounts.dev
CLERK_AUTHORIZED_PARTIES=http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8501,http://localhost:8501
CLERK_JWT_TEMPLATE=rag-api
STREAMLIT_URL=http://127.0.0.1:8501
CORS_ALLOWED_ORIGINS=http://127.0.0.1:8501
STUDENT_DOMAINS=charusat.edu.in,edu.charusat.org
FACULTY_DOMAINS=charusat.org,charusat.ac.in
ADMIN_EMAILS=admin@charusat.org
CLERK_ROLE_CACHE_TTL_SECONDS=60
```

Never commit `.env` or Clerk keys.

## Clerk dashboard

1. Create a free Clerk application.
2. Configure the sign-in methods and allowed email/domain restrictions in **User & authentication**.
3. Add a JWT template named `rag-api` under **JWT Templates**.
4. Configure the template to include the issuer and public metadata claims. A minimal template payload is:

```json
{
  "sub": "{{user.id}}",
  "email": "{{user.primary_email_address}}",
  "public_metadata": {{user.public_metadata}},
  "azp": "{{request.origin}}"
}
```

Use the exact issuer and authorized-party values issued by your Clerk instance. The backend verifies `exp`, `iss`, `azp`, the RS256 algorithm, and the signing key from Clerk JWKS.

5. Use one local host consistently: this project uses `127.0.0.1`. Add these development URLs wherever Clerk asks for application URLs or allowed origins:
   - `http://127.0.0.1:8501` (Streamlit application)
   - `http://127.0.0.1:8000/auth/clerk` (FastAPI Clerk bridge)
   - `http://127.0.0.1:8000` (FastAPI origin, when an origin is requested)
6. In Clerk's redirect settings, set sign-in and sign-up fallback/force redirect URLs to `http://127.0.0.1:8000/auth/clerk`. The bridge exchanges the Clerk session for the `rag-api` JWT and then sends the browser to `http://127.0.0.1:8501`.
7. If using custom Microsoft OAuth credentials, Azure's redirect URI must be Clerk's callback URI, not the local app URL: `https://vital-marten-9147.clerk.accounts.dev/v1/oauth_callback`. Clerk's shared development credentials are sufficient for local testing.
8. Set new users to `student` by default. The backend then resolves the effective role from the verified primary email on every protected request. `public_metadata.role` is synchronized automatically and is not trusted as the source of truth.
9. Put the exact administrator email addresses in `ADMIN_EMAILS`. They must also use one of the configured faculty domains.
10. Create a Clerk `user.created` webhook pointing to `/webhooks/clerk`, add the signing secret as `CLERK_WEBHOOK_SIGNING_SECRET`, and keep the webhook on HTTPS in production. The webhook writes `{"role":"student"}` to new users’ public metadata.

## CHARUSAT membership policy

The backend accepts only verified primary emails from the configured domains:

- `charusat.edu.in`, `edu.charusat.org` -> `student`
- `charusat.org`, `charusat.ac.in` -> `faculty`
- an exact email in `ADMIN_EMAILS` with a faculty domain -> `admin`

Lookalike domains are rejected. Users outside these domains receive `You are not a CHARUSAT Member`, are audited, and are deleted through the Clerk Backend API when possible. Clerk API failures fail closed.

## Assign roles

```bash
uv run python scripts/assign_clerk_role.py user_abc123 faculty --department cs
uv run python scripts/assign_clerk_role.py user_abc123 admin
```

The admin API endpoint is `PUT /admin/roles/{clerk_user_id}` and requires an authenticated admin token.

## Authentication flow

The Streamlit sidebar links to `GET /auth/clerk`. That page uses Clerk's hosted sign-in UI with explicit `signInForceRedirectUrl` and `signUpForceRedirectUrl` values pointing back to `/auth/clerk`, obtains a short-lived token from the `rag-api` template, and redirects to Streamlit. If an OAuth provider falls back to the FastAPI root with `__clerk_*` parameters, FastAPI forwards those parameters to Streamlit and Streamlit sends them back through the Clerk bridge. Streamlit stores the token in session state and sends it as `Authorization: Bearer ...` to FastAPI. A `401` clears the session and requires a new sign-in.

For production, put FastAPI and Streamlit behind one HTTPS domain and use HTTPS-only deployment URLs. The local query-parameter handoff is intended for the free-tier local setup; the token is immediately removed from Streamlit’s query parameters after capture.

## Qdrant access model

Every new point contains:

```json
{
  "visibility": "public | student | faculty | admin",
  "owner_id": "user_xxx",
  "department": "cs",
  "doc_id": "document-id"
}
```

The backend builds the Qdrant filter from verified Clerk claims. Students receive only `public` and `student` points. Faculty receive those plus `faculty` points in their department or owned by them. Admins receive all points. The filter is applied inside Qdrant before context reaches the LLM.

## Existing-data migration

1. Back up the Qdrant collection and SQLite/Postgres database.
2. Review existing document owners and departments.
3. Run:

```bash
uv run python migrate_rbac.py --legacy-private-visibility faculty
```

4. Assign correct Clerk IDs and departments to document records where the old local email cannot be mapped automatically.
5. Re-ingest documents whose points do not have reliable metadata. Re-ingestion replaces the source’s points and is the preferred cleanup path.
6. Verify with a student token that faculty/admin-only source IDs do not appear in `GET /sources` and cannot be retrieved by `POST /ask`.

The migration maps legacy `private` to `faculty` by default. Use `--legacy-private-visibility admin` when the old private collection must remain administrator-only.

## Permission matrix

| Endpoint | Admin | Faculty | Student |
|---|---:|---:|---:|
| `POST /upload`, `POST /ingest` | Allow | Allow, own department | Deny |
| `DELETE /documents/{id}` | Allow | Own document/department only | Deny |
| `POST /ask` | All sources | Public/student plus department/owned faculty docs | Public/student only |
| `GET /sources` | All | Accessible sources | Public/student sources |
| `GET /preview/{id}/{page}` | Accessible | Accessible | Public/student only |
| `POST /export` | Allow | Allow | Allow |
| `GET /admin/users` | Allow | Deny | Deny |
| `PUT /admin/roles/{id}` | Allow | Deny | Deny |
| `GET /admin/logs` | Allow | Deny | Deny |

Unauthenticated requests receive `401`; authenticated users with insufficient role or scope receive `403`.
