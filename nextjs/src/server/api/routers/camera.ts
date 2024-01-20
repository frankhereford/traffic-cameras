import { z } from "zod";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const cameraRouter = createTRPCRouter({
  setStatus: protectedProcedure
    .input(
      z.object({
        cameraId: z.number(),
        status: z.string(),
        hex: z.string().optional(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      // console.log("input", input);

      let camera = await ctx.db.camera.findFirst({
        where: {
          cameraId: input.cameraId,
          user: { id: ctx.session.user.id },
        },
      });

      if (!camera) {
        camera = await ctx.db.camera.create({
          data: {
            cameraId: input.cameraId,
            user: { connect: { id: ctx.session.user.id } },
          },
        });
      }

      let status = await ctx.db.status.findFirst({
        where: {
          slug: input.status,
        },
      });

      if (!status) {
        status = await ctx.db.status.create({
          data: {
            name: input.status,
            slug: input.status.toLowerCase().replace(/\s+/g, "-"),
          },
        });
      }

      if (input.hex) {
        console.log("\n\ninput.hex", input.hex);
      }

      const instance = await ctx.db.instance.create({
        data: {
          cameraId: camera.id,
          hex: input.hex,
        },
      });

      camera = await ctx.db.camera.update({
        where: {
          id: camera.id,
        },
        data: {
          status: { connect: { id: status.id } },
        },
        include: {
          status: true,
        },
      });

      return instance;
    }),
  get: protectedProcedure.input(z.object({})).query(({ ctx, input }) => {
    console.log("input:", input);
  }),
});
