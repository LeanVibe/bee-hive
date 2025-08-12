#!/usr/bin/env node
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { exec as _exec } from 'node:child_process'
import { promisify } from 'node:util'
const exec = promisify(_exec)

async function main() {
  const tempDir = await mkdtemp(join(tmpdir(), 'ws-schema-'))
  const tempOut = join(tempDir, 'ws-messages.d.ts')
  try {
    await exec(`npx json-schema-to-typescript -i ../schemas/ws_messages.schema.json -o ${tempOut}`)
    const [current, generated] = await Promise.all([
      readFile(new URL('../src/types/ws-messages.d.ts', import.meta.url), 'utf8'),
      readFile(tempOut, 'utf8'),
    ])
    if (current.trim() !== generated.trim()) {
      console.error('Schema/type drift detected between schemas/ws_messages.schema.json and src/types/ws-messages.d.ts')
      process.exitCode = 2
    } else {
      console.log('Schema parity OK')
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
