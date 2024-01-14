import { z } from "zod"
import * as fs from "fs"
import * as os from "os"
import * as path from "path"
import { v4 as uuidv4 } from "uuid"

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc"

export const transformation = createTRPCRouter({
  submitWarpRequest: publicProcedure
    .input(
      z.object({
        image: z.string(),
        points: z.array(
          z.object({
            cctvPoint: z.object({
              x: z.number(),
              y: z.number(),
            }),
            mapPoint: z.object({
              lat: z.number(),
              lng: z.number(),
            }),
          }),
        ),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // console.log("input", JSON.stringify(input, null, 2))

      const baseDir = path.join(os.tmpdir(), "transformations")
      if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir)
      }

      const uuid = uuidv4()
      const tmpDir = path.join(baseDir, uuid)
      fs.mkdirSync(tmpDir, { recursive: true })

      console.log("Temporary directory created.", tmpDir)

      // Remove the base64 image prefix
      const base64Data = input.image.replace(/^data:image\/jpeg;base64,/, "")

      // Convert the base64 string to a Buffer
      const dataBuffer = Buffer.from(base64Data, "base64")

      // Write the Buffer to a file
      const imagePath = path.join(tmpDir, "cctvImage.jpg")
      fs.writeFileSync(imagePath, dataBuffer)
      // console.log("Image written to disk at", imagePath)

      console.log("done")

      return uuid
    }),

  getSecretMessage: protectedProcedure.query(() => {
    return "you can now see this secret message!"
  }),
})
