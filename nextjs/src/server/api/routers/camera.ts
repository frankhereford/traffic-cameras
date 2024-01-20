import { z } from "zod";

import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";

export const cameraRouter = createTRPCRouter({
  setStatus: protectedProcedure
    .input(z.object({ cameraId: z.number(), status: z.string() }))
    .mutation(async ({ ctx, input }) => {
      console.log("input", input);

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
      console.log("cameraId: ", camera.cameraId);

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

      if (status) {
        console.log("status: ", status);
      }

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

      return camera;
    }),
});
